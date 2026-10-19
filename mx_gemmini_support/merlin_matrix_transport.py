"""Checked BF16 matrix readout and VPU consumer entries for private edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def register_matrix_transport(generated, shape, symbol, issuer_symbol, fixed_symbols,
                              workspace_symbol, compile_source, abi_entries, *,
                              prepared_activation=False, retain_prepared_rhs=False):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .dense_kernel import emit_dense_f32_kernel

    suffix = "_prepared_bf16_transport" if prepared_activation else "_bf16_transport"
    entry, kernel = symbol + suffix, symbol + suffix + "_kernel"
    adapter = emit_dense_f32_kernel(
        generated.issuer, shape, symbol=kernel, issuer_symbol=issuer_symbol,
        shared_workspace_symbol=workspace_symbol, prepacked_weight_symbols=fixed_symbols,
        output_dtype="bf16", prepared_activation=prepared_activation,
        retain_prepared_rhs=retain_prepared_rhs)
    compile_source(kernel, adapter.source)
    m, n, k = shape
    packed = generated.issuer.buffer_bytes["activation"] + generated.issuer.buffer_bytes["activation_scales"]
    buffers = [LogicalBuffer("lhs", "input", "input_0", (packed,) if prepared_activation else (m, k),
                             "i8" if prepared_activation else "f32")]
    input_types = [f"tensor<{packed}xi8>" if prepared_activation else f"tensor<{m}x{k}xf32>"]
    if not prepared_activation or retain_prepared_rhs:
        buffers.append(LogicalBuffer("unused_rhs", "input", "input_1", (k, n), "f32"))
        input_types.append(f"tensor<{k}x{n}xf32>")
    buffers.append(LogicalBuffer("out", "output", "out_0", (m, n), "bf16"))
    abi_entries[entry] = (kernel, tuple(buffers))
    return {"symbol": entry, "kernel_symbol": kernel,
            "workspace_bytes": adapter.workspace_bytes,
            "original_argument_types": [*input_types, f"tensor<{m}x{n}xf32>"],
            "direct_readout": (m, n) == generated.issuer.shape_mnk[:2],
            "completion": "synchronous_native_issuer",
            "rounding": "unchanged_matrix_bf16_readout"}


def register_add_transports(site, profile, compile_source, abi_entries):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .vpu_add_kernel import emit_vpu_elementwise

    if site["dtype"] != "f32":
        return []
    records = []
    for index in (0, 1):
        entry = site["symbol"] + f"_bf16_input{index}"
        kernel = entry + "_kernel"
        dtypes = ["bf16" if i == index else "f32" for i in (0, 1)]
        generated = emit_vpu_elementwise(
            profile, result_shape=tuple(site["shape"]), rhs_shape=tuple(site["rhs_shape"]),
            dtype="f32", operation="add", symbol=kernel,
            lhs_dtype=dtypes[0], rhs_dtype=dtypes[1], preserve_f32_nan_refusal=True)
        compile_source(kernel, generated.source)
        abi_entries[entry] = (kernel, (
            LogicalBuffer("lhs", "input", "input_0", tuple(site["shape"]), dtypes[0]),
            LogicalBuffer("rhs", "input", "input_1", tuple(site["rhs_shape"]), dtypes[1]),
            LogicalBuffer("out", "output", "out_0", tuple(site["shape"]), "f32")))
        records.append({"symbol": entry, "kernel_symbol": kernel, "input_index": index,
                        "numeric_policy": "preserve_f32_boundary_nan_refusal"})
    return records


class MXMatrixTransportPostOffload:
    """Delegate private-edge legality to Merlin using exact catalog bindings."""

    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from merlin.xdsl_dialects.lowering.private_transport import (
            PrivateTensorTransport,
            rewrite_private_transports,
        )
        from xdsl.dialects.builtin import bf16

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(sidecar.read_text())
        if (Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
                or routing["model_sha256"] != catalog["source_sha256"]
                or catalog.get("matrix_vpu_transport") != "experimental_private_bf16"):
            raise ValueError("matrix transport catalog differs from source routing")
        bindings = {r["region"]: r for r in catalog["bindings"]}
        if len(bindings) != len(catalog["bindings"]) or len(routing["routed"]) != len(bindings):
            raise ValueError("matrix transport source binding count changed")
        entries = {}
        kernels = {k["symbol"]: k for k in catalog["kernels"] if k.get("private_bf16_transport")}
        for row in routing["routed"]:
            bound = bindings.get(row["source_region"])
            if (bound is None or bound["source_operation_ordinal"] != row["source_operation_ordinal"]
                    or bound["tensor_types"] != row["tensor_types"]):
                raise ValueError("matrix transport source binding changed")
            if bound["symbol"] in kernels:
                kernel = kernels[bound["symbol"]]
                endpoint = (kernel, kernel["private_bf16_transport"])
                if entries.setdefault(row["symbol"], endpoint) != endpoint:
                    raise ValueError("matrix transport call has inconsistent bindings")
                if kernel.get("prepared_private_bf16_transport"):
                    name = kernel["activation_preparation"]["consumer_symbol"]
                    endpoint = (kernel, kernel["prepared_private_bf16_transport"])
                    if entries.setdefault(name, endpoint) != endpoint:
                        raise ValueError("prepared matrix transport binding changed")
        consumers = {k["symbol"]: {v["input_index"]: v for v in k.get("private_bf16_inputs", [])}
                     for k in catalog["vpu_add"]["kernels"]}

        def select(producer, consumer, index):
            entry = entries.get(producer.callee.root_reference.data)
            endpoint = consumers.get(consumer.callee.root_reference.data, {}).get(index)
            if entry is None or endpoint is None:
                return None
            kernel, transport = entry
            if (kernel["batched"] or not kernel["prepacked_checkpoint_weight"]
                    or [str(v.type) for v in producer.arguments] != transport["original_argument_types"]):
                raise ValueError("matrix transport producer shape or immutable weight changed")
            return PrivateTensorTransport(transport["symbol"],
                                           endpoint["symbol"], bf16)

        result, edges = rewrite_private_transports(parse_mlir_file(prepared), select)
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_matrix_transport.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (workdir / "mx_matrix_transport_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.private_matrix_transport.v1",
            "source_sha256": digest(prepared), "output_sha256": digest(output),
            "catalog_sha256": digest(self.catalog_path), "sidecar_sha256": digest(sidecar),
            "selection": "experimental_private_edge_legality_only", "edges": edges,
        }, indent=2, sort_keys=True) + "\n")
        return output
