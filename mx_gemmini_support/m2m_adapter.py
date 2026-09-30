"""Installed model2MLIR adapter for the selected MX software contract."""

from __future__ import annotations

import hashlib
import json
import types

import torch
from torch import nn

from .contract import compile_contract, contract_digest
from .policy import load_policy
from .torchao_quant import (MXGemminiFakeQuantConfig, MXGemminiLinear, _dequant_operand_for_graph,
                            expose_sdpa_contractions,
                            quantize_functional_contractions_, verify_kernel_contract)


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _freeze_eval(graph_module):
    try:
        from torch.ao.quantization import allow_exported_model_train_eval
        allow_exported_model_train_eval(graph_module)
    except (ImportError, AttributeError):
        def train(self, mode=True):
            if mode:
                raise ValueError("MX exported graph is inference only")
            self.training = False
            return self
        graph_module.train = types.MethodType(train, graph_module)
        graph_module.eval = types.MethodType(lambda self: self.train(False), graph_module)


def _linear_call_shapes(exported, module_names):
    """Observe static Linear calls before TorchAO replaces their modules."""
    calls = {name: [] for name in module_names}
    for node in exported.graph_module.graph.nodes:
        if node.target != torch.ops.aten.linear.default:
            continue
        stack = node.meta.get("nn_module_stack") or {}
        owners = {row[0] for row in stack.values()
                  if isinstance(row, (tuple, list)) and row and row[0] in calls}
        if len(owners) > 1:
            raise ValueError(f"MX Linear call has ambiguous module owners: {sorted(owners)}")
        if not owners:
            continue
        owner = owners.pop()
        value = node.args[0].meta.get("val") if node.args else None
        shape = getattr(value, "shape", None)
        calls[owner].append(tuple(shape) if shape is not None else None)
    return calls


def _rewrite_linear_nodes(graph_module, selections, quantized_model):
    """Apply the TorchAO handler's static weights on source-stamped Linear calls."""
    selected = {name: (fmt, books) for name, _module, fmt, books in selections}
    seen = set()
    graph = graph_module.graph
    for node in tuple(graph.nodes):
        if node.target != torch.ops.aten.linear.default:
            continue
        stack = node.meta.get("nn_module_stack") or {}
        owners = {row[0] for row in stack.values()
                  if isinstance(row, (tuple, list)) and row and row[0] in selected}
        if not owners:
            continue
        if len(owners) != 1:
            raise ValueError("MX Linear call has ambiguous selected module owners")
        name = owners.pop()
        if name in seen:
            raise ValueError(f"MX Linear {name!r} has multiple static calls")
        seen.add(name)
        fmt, books = selected[name]
        transformed = quantized_model.get_submodule(name)
        if not isinstance(transformed, MXGemminiLinear):
            raise TypeError(f"TorchAO did not replace selected Linear {name!r}")
        activation, weight, *bias = node.args
        if (not isinstance(weight, torch.fx.Node) or weight.op != "get_attr"
                or weight.target != f"{name}.weight"):
            raise ValueError(f"MX Linear {name!r} has an unsupported exported weight binding")
        parameter = graph_module.get_parameter(weight.target)
        if parameter.shape != transformed.weight.shape:
            raise ValueError(f"MX Linear {name!r} weight shape changed during capture")
        owner_name, _, local_name = weight.target.rpartition(".")
        owner = graph_module.get_submodule(owner_name) if owner_name else graph_module
        owner.register_parameter(local_name, nn.Parameter(
            transformed.weight.detach().clone(), requires_grad=False))
        dtype = getattr(activation.meta.get("val"), "dtype", None)
        if dtype is None:
            raise ValueError(f"MX Linear {name!r} activation dtype is unobserved")
        lineage = dict(node.meta.get("custom") or {})
        with graph.inserting_before(node):
            q_activation = graph.call_function(
                _dequant_operand_for_graph, args=(activation, fmt, -1, books[0]))
            bf16_activation = graph.call_function(
                torch.ops.aten.to.dtype, args=(q_activation, torch.bfloat16))
            bf16_bias = None
            if bias and isinstance(bias[0], torch.fx.Node):
                bf16_bias = graph.call_function(
                    torch.ops.aten.to.dtype, args=(bias[0], torch.bfloat16))
        node.args = (bf16_activation, weight, bf16_bias)
        with graph.inserting_after(node):
            result = graph.call_function(torch.ops.aten.to.dtype, args=(node, dtype))
        result.meta["val"] = node.meta.get("val")
        for inserted in (q_activation, bf16_activation, bf16_bias, result):
            if inserted is not None:
                inserted.meta["custom"] = dict(lineage)
        node.replace_all_uses_with(result)
        result.args = (node, dtype)
    if seen != selected.keys():
        raise ValueError(f"selected MX Linear calls are missing from source graph: {sorted(selected.keys() - seen)}")
    graph.lint()
    graph_module.recompile()


def _verify_chain_edges(graph_module, chains):
    """A resident output may only feed the next contraction directly as A."""
    nodes = {node.name: node for node in graph_module.graph.nodes}
    for producer, chain in chains.items():
        consumer_id = chain["consumer"]
        if not consumer_id.startswith("functional:"):
            raise ValueError("resident output chains currently require a visible functional consumer")
        consumer = nodes.get(consumer_id.removeprefix("functional:"))
        if consumer is None or not consumer.args:
            raise ValueError("resident output consumer is absent from exported graph")
        lhs = consumer.args[1 if consumer.target == torch.ops.aten.addmm.default else 0]
        if (getattr(lhs, "target", None) == torch.ops.aten.to.dtype
                and len(lhs.args) >= 2 and lhs.args[1] == torch.float32
                and getattr(lhs.args[0].meta.get("val"), "dtype", None) == torch.bfloat16):
            # The fake-quant wrapper widens its BF16 value for host PyTorch.
            # This exact widening changes no value and need not be a host seam.
            lhs = lhs.args[0]
        if producer.startswith("functional:"):
            source = nodes.get(producer.removeprefix("functional:"))
            if lhs is not source:
                raise ValueError("resident output does not directly feed consumer A")
        elif producer.startswith("module:"):
            fqn = producer.removeprefix("module:")
            stack = getattr(lhs, "meta", {}).get("nn_module_stack") or {}
            matches = any(isinstance(row, (tuple, list)) and row and row[0] == fqn
                          for row in stack.values())
            if getattr(lhs, "target", None) != torch.ops.aten.linear.default or not matches:
                raise ValueError("resident Linear output does not directly feed consumer A")
        else:
            raise ValueError("resident output producer has an invalid site ID")


def apply(model, inputs, *, contract_bytes, policy_bytes, original_frontend_snapshot=None):
    """Return a Q/DQ graph and a complete static contraction census.

    The graph is a capture diagnostic. Its BF16 matmuls are not RTL products.
    """
    from torchao.quantization import quantize_

    if not isinstance(model, nn.Module):
        raise TypeError("MX adapter requires a torch.nn.Module")
    contract = compile_contract(contract_bytes)
    verify_kernel_contract(contract)
    policy = load_policy(policy_bytes)
    if policy.graph_sha256 is not None:
        if not isinstance(original_frontend_snapshot, dict) or original_frontend_snapshot.get("status") != "complete":
            raise ValueError("functional overrides require a complete prequantization graph snapshot")
        if original_frontend_snapshot.get("sha256") != policy.graph_sha256:
            raise ValueError("functional policy source graph sha256 differs from captured model")
    from m2m.capture.trace import (attach_original_identity,
                                   prepare_lifted_constant_lineage,
                                   snapshot_exported_program)

    exported = torch.export.export(model.eval(), tuple(inputs))
    if original_frontend_snapshot is not None:
        actual = snapshot_exported_program(exported, stage="quantization_input")
        if not attach_original_identity(exported, original_frontend_snapshot, actual):
            raise ValueError("selected MX source snapshot differs from prequantization export")
        source_graph_sha256 = original_frontend_snapshot["sha256"]
    else:
        source_graph_sha256 = snapshot_exported_program(exported, stage="original")["sha256"]
    census = []
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    unknown_modules = set(policy.module_overrides) - {name for name, _ in modules}
    if unknown_modules:
        raise ValueError(f"MX policy names missing Linear modules: {sorted(unknown_modules)}")
    call_shapes = _linear_call_shapes(exported, {name for name, _ in modules})
    selections = []
    for name, module in modules:
        fmt = policy.module_format(name)
        site_id = f"module:{name}"
        observed = call_shapes[name]
        if len(observed) > 1:
            raise ValueError(f"MX Linear {name!r} has multiple static calls; module site IDs cannot distinguish them")
        if not observed:
            census.append({"site_id": site_id, "kind": "linear", "status": "skipped",
                           "reason": "module absent from exported graph"})
            continue
        if fmt == "host":
            census.append({"site_id": site_id, "kind": "linear", "status": "host"})
            continue
        bounds = contract["formats"][fmt]["shape_bounds"]
        n, k = module.out_features, module.in_features
        if k < bounds["K"]["min"] or k % bounds["K"]["multiple_of"] or n < bounds["N"]["min"] or n % bounds["N"]["multiple_of"]:
            census.append({"site_id": site_id, "kind": "linear", "status": "skipped",
                           "reason": f"N={n} K={k} outside {fmt} shape bounds"})
            continue
        shape = observed[0]
        if (shape is None or len(shape) not in (2, 3, 4)
                or any(not isinstance(dim, int) for dim in shape[-2:])):
            census.append({"site_id": site_id, "kind": "linear", "status": "skipped",
                           "reason": "unknown or unsupported activation rank/shape"})
            continue
        m, observed_k = shape[-2:]
        if observed_k != k:
            raise ValueError(f"MX Linear {name!r} captured K differs from module weight K")
        if m < bounds["M"]["min"] or m % bounds["M"]["multiple_of"]:
            census.append({"site_id": site_id, "kind": "linear", "status": "skipped",
                           "reason": f"M={m} outside {fmt} shape bounds"})
            continue
        books = policy.codebooks(site_id) if fmt == "mxfp6" else (None, None)
        selections.append((name, module, fmt, books))
        census.append({"site_id": site_id, "kind": "linear", "status": "quantized",
                       "format": fmt, "shape": [m, n, k],
                       "fp6_codebook_sha256": _sha(books) if fmt == "mxfp6" else None})
    for name, module, fmt, books in selections:
        quantize_(model, MXGemminiFakeQuantConfig(fmt, *books),
                  filter_fn=lambda candidate, fqn, wanted=name, selected=module:
                  fqn == wanted and candidate is selected)
    exported = expose_sdpa_contractions(exported)
    graph_module = exported.module()
    _verify_chain_edges(graph_module, policy.output_chains)
    _rewrite_linear_nodes(graph_module, selections, model)
    seen_functional = set()

    def select(site_id):
        seen_functional.add(site_id)
        fmt = policy.functional_format(site_id)
        books = policy.codebooks(site_id) if fmt == "mxfp6" else (None, None)
        return fmt, books

    functional = quantize_functional_contractions_(
        graph_module, select, frozenset(name for name, _ in modules))
    unknown_functional = set(policy.functional_overrides) - seen_functional
    if unknown_functional:
        raise ValueError(f"MX policy names missing functional sites: {sorted(unknown_functional)}")
    for site in functional:
        if site.get("format") == "mxfp6":
            site["fp6_codebook_sha256"] = _sha(policy.codebooks(site["site_id"]))
    census.extend(functional)
    by_site = {site["site_id"]: site for site in census}
    for producer, chain in policy.output_chains.items():
        source = by_site.get(producer)
        consumer = by_site.get(chain["consumer"])
        if source is None or consumer is None or source["status"] != "quantized" or consumer["status"] != "quantized":
            raise ValueError(f"output chain {producer!r} needs quantized producer and consumer")
        if chain["consumer"] == producer or consumer["format"] != chain["format"]:
            raise ValueError(f"output chain {producer!r} has incompatible consumer format")
        source_n = source["shape"][-2]
        consumer_k = consumer["shape"][-1]
        if source_n != consumer_k:
            raise ValueError(f"output chain {producer!r} output N differs from consumer K")
        if len(source["shape"]) == 3 and source["shape"][0] != consumer["shape"][0]:
            raise ValueError(f"output chain {producer!r} output M differs from consumer M")
        if chain["format"] == "mxfp6":
            output_book = policy.output_codebook(producer)
            if output_book != policy.codebooks(chain["consumer"])[0]:
                raise ValueError("resident FP6 output LUT differs from consumer activation LUT")
            source["output_fp6_codebook_sha256"] = _sha(output_book)
        source["output_chain"] = chain
    if not census:
        raise ValueError("MX adapter found no contraction sites")
    _freeze_eval(graph_module)
    prepare_lifted_constant_lineage(graph_module)
    captured = torch.export.export(graph_module, tuple(inputs))
    manifest = {
        "schema": "m2m.quantization_manifest.v1",
        "adapter_id": "mx_gemmini",
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "policy_sha256": policy.source_sha256,
        "compiled_contract_sha256": contract_digest(contract),
        "source_graph_sha256": source_graph_sha256,
        "numeric_status": "operand_fake_quant_only",
        "sites": census,
    }
    return captured, manifest
