"""Bind source-proved repeated RHS batches to qualified private MX preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_bound_repetitions(prepared, plan):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.repeated_batch_input import (
        prove_repeated_batch_input,
        validate_repeated_batch_input,
    )

    prepared = Path(prepared)
    if _sha(prepared) != plan["source_sha256"]:
        raise ValueError("batch reuse source changed from the original catalog plan")
    module = parse_mlir_file(prepared)
    module.verify()
    operations = tuple(module.walk())
    kernels = {entry["symbol"]: entry for entry in plan["kernels"]}
    selected = {}
    for index, binding in enumerate(plan["bindings"]):
        if kernels[binding["symbol"]]["rank"] != 3:
            continue
        ordinal = binding["source_operation_ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal < len(operations):
            raise ValueError("batch reuse source ordinal changed")
        operation = operations[ordinal]
        if (len(operation.operands) != 3 or len(operation.results) != 1
                or getattr(operation.attributes.get("prov.region_id"), "data", None) != binding["region"]
                or [str(v.type) for v in (*operation.operands[:2], operation.results[0])] != binding["tensor_types"]):
            raise ValueError("batch reuse source binding differs from original catalog types")
        try:
            proof = prove_repeated_batch_input(operation.operands[1])
        except ValueError:
            continue
        validate_repeated_batch_input(proof)
        selected[index] = proof.to_dict()
    if _sha(prepared) != plan["source_sha256"]:
        raise ValueError("batch reuse source changed during analysis")
    return selected


def register_repetitions(entry, generated, bindings, selected, workspace,
                         compile_source, abi_entries, *, virtual_zero_padding=False, compact_rhs=False):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .dense_kernel import emit_dense_f32_batch_kernel

    compiled = {}
    for index, proof in selected.items():
        binding = bindings[index]
        if binding["symbol"] != entry["symbol"]:
            continue
        batch, m, n, k = entry["shape"]
        repeat = proof["contiguous_batch_repeat"]
        for compact_input in ((False, True) if compact_rhs else (False,)):
            logical_entry = f"mx_{'compact_' if compact_input else ''}repeated_rhs_{index}"
            kernel = f"{entry['symbol']}_repeated_rhs_{repeat}_{'compact_' if compact_input else ''}kernel"
            if kernel not in compiled:
                alternative = emit_dense_f32_batch_kernel(
                    generated.issuer, (batch, m, n, k), symbol=kernel,
                    issuer_symbol=entry["issuer_symbol"], shared_workspace_symbol=workspace,
                    virtual_zero_padding=virtual_zero_padding, rhs_batch_repeat=repeat, compact_rhs=compact_input)
                if alternative.workspace_bytes != generated.kernel.workspace_bytes:
                    raise ValueError("batch reuse changed the original serial workspace extent")
                compile_source(kernel, alternative.source)
                compiled[kernel] = alternative.workspace_bytes
            abi_entries[logical_entry] = (kernel, (
                LogicalBuffer("lhs", "input", "input_0", (batch, m, k), "f32"),
                LogicalBuffer("rhs", "input", "input_1", (batch // repeat if compact_input else batch, k, n), "f32"),
                LogicalBuffer("out", "output", "out_0", (batch, m, n), "f32")))
            binding["batch_rhs_compact" if compact_input else "batch_rhs_reuse"] = {
                "entry_symbol": logical_entry, "kernel_symbol": kernel,
                "repeat": repeat, "source_proof": proof,
                "completion": "synchronous_native_issuer",
                "lifetime": "private_serial_batch_call",
                "workspace_bytes": compiled[kernel],
                "provider_obligation": "original_results_rounding_effects_and_partial_refusal",
                **({"compact_input": True} if compact_input else {}),
            }


class MXBatchRHSReusePostOffload:
    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.llvmlower.repeated_batch_input import (
            RepeatedBatchConsumer,
            private_repeated_batch_inputs,
            prove_repeated_batch_input,
            rewrite_repeated_batch_consumers,
        )
        from merlin.xdsl_dialects._common import text
        from xdsl.dialects.func import CallOp

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(sidecar.read_text())
        compact = catalog.get("compact_batch_rhs", False)
        if type(compact) is not bool:
            raise ValueError("compact batch RHS requires an explicit boolean ABI selection")
        if compact and catalog.get("compact_profitability_guard") != "private_materialization":
            raise ValueError("compact input ABI requires a private-materialization guard and ordinary reuse fallback")
        if (Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
                or routing["model_sha256"] != catalog["source_sha256"]
                or catalog.get("batch_rhs_reuse") != "experimental_exact_contiguous_repetition"
                or catalog.get("integer_group_packing") is not True
                or _sha(routing["catalog_object"]) != catalog["compilation"]["object_sha256"]):
            raise ValueError("batch RHS reuse differs from selected source/catalog/object routing")
        bindings = {binding["region"]: binding for binding in catalog["bindings"]}
        if (len(bindings) != len(catalog["bindings"]) or len(bindings) != len(routing["routed"])
                or len({row["source_operation_ordinal"] for row in catalog["bindings"]}) != len(bindings)
                or {row["source_region"] for row in routing["routed"]} != set(bindings)):
            raise ValueError("batch RHS source binding coverage changed")
        contracts, verified, families = [], set(), {}
        for row in routing["routed"]:
            binding = bindings.get(row["source_region"])
            if (binding is None or row["source_operation_ordinal"] != binding["source_operation_ordinal"]
                    or row["tensor_types"] != binding["tensor_types"]):
                raise ValueError("batch RHS source binding type or ordinal changed")
            family = families.setdefault(row["symbol"], {
                "kernel": binding["symbol"], "types": binding["tensor_types"], "source_calls": 0, "alternatives": {}})
            if family["kernel"] != binding["symbol"] or family["types"] != binding["tensor_types"]:
                raise ValueError("shared batch declaration binds incompatible native kernel families")
            family["source_calls"] += 1
            ordinary, compact_endpoint = binding.get("batch_rhs_reuse"), binding.get("batch_rhs_compact")
            if compact_endpoint is not None and ordinary is None:
                raise ValueError("compact input ABI lacks its qualified ordinary reuse fallback")
            if compact and ordinary is not None and (compact_endpoint is None
                    or compact_endpoint["source_proof"] != ordinary["source_proof"]
                    or compact_endpoint["workspace_bytes"] != ordinary["workspace_bytes"]):
                raise ValueError("compact input ABI lacks a matching qualified ordinary reuse fallback")
            for alternative, compact_input in [(ordinary, False), (compact_endpoint, True)]:
                if alternative is None:
                    continue
                if alternative.get("compact_input", False) is not compact_input or compact_input and not compact:
                    raise ValueError("batch RHS endpoint differs from the selected compact input ABI")
                if (alternative["completion"] != "synchronous_native_issuer"
                        or alternative["lifetime"] != "private_serial_batch_call"
                        or alternative["provider_obligation"] != "original_results_rounding_effects_and_partial_refusal"):
                    raise ValueError("batch RHS preparation lifetime or completion changed")
                repeat = alternative["repeat"]
                source_proof = alternative["source_proof"]
                if (type(repeat) is not int or repeat <= 1
                        or source_proof["contiguous_batch_repeat"] != repeat
                        or source_proof["shape"][0] % repeat):
                    raise ValueError("batch RHS alternative changed its original source repetition")
                kernel = alternative["kernel_symbol"]
                if not kernel.isascii() or not kernel.isidentifier():
                    raise ValueError("batch RHS compiled kernel binding is invalid")
                if kernel not in verified:
                    if _sha(self.catalog_path.parent / (kernel + ".c")) != catalog["compilation"]["source_sha256"][kernel]:
                        raise ValueError("batch RHS compiled kernel source changed")
                    verified.add(kernel)
                previous = family["alternatives"].setdefault((repeat, compact_input), alternative)
                if previous["kernel_symbol"] != kernel or previous["workspace_bytes"] != alternative["workspace_bytes"]:
                    raise ValueError("shared batch repetition has inconsistent compiled endpoints")
        result = parse_mlir_file(prepared)
        proved, compact_candidates = {}, {}
        for symbol, family in families.items():
            if not family["alternatives"]:
                continue
            calls = [op for op in result.walk() if isinstance(op, CallOp)
                     and op.callee.root_reference.data == symbol]
            if len(calls) != family["source_calls"]:
                raise ValueError("shared batch declaration call coverage changed")
            for call in calls:
                if [str(value.type) for value in call.arguments] != family["types"]:
                    raise ValueError("shared batch call changed its original typed arguments")
                try:
                    proof = prove_repeated_batch_input(call.arguments[1])
                except ValueError:
                    continue
                if (proof.repeat, False) in family["alternatives"]:
                    proved[call] = (symbol, proof)
                    if compact and (proof.repeat, True) in family["alternatives"]:
                        compact_candidates[(call, 1)] = proof
        private = private_repeated_batch_inputs(compact_candidates) if compact else frozenset()
        grouped = {}
        for call, (symbol, proof) in proved.items():
            grouped.setdefault((symbol, proof.repeat, (call, 1) in private), []).append(call)
        for (symbol, repeat, compact_input), selected in grouped.items():
            alternative = families[symbol]["alternatives"][(repeat, compact_input)]
            contracts.append(RepeatedBatchConsumer(symbol, alternative["entry_symbol"], 1, repeat,
                                                    tuple(selected), compact=compact_input))
        report = rewrite_repeated_batch_consumers(result, contracts)
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_batch_rhs_reuse.mlir"
        output.write_text(text(result))
        (workdir / "mx_batch_rhs_reuse_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.batch_rhs_reuse.v1", "source_sha256": _sha(prepared),
            "output_sha256": _sha(output), "catalog_sha256": _sha(self.catalog_path),
            "sidecar_sha256": _sha(sidecar), "selection": "experimental_source_proof_and_qualified_serial_endpoint",
            "consumers": report, "selected_calls": sum(row["calls"] for row in report),
            **({"compaction_cost_guard": "private_materialization",
                "compact_calls": sum(row["calls"] for row in report if row.get("compact_input")),
                "ordinary_reuse_fallback_calls": sum(row["calls"] for row in report if not row.get("compact_input"))}
               if compact else {}),
        }, indent=2) + "\n")
        return output
