"""Find checkpoint-backed MX RHS operands eligible for offline packing.

This audits a compiled stage, after Merlin has routed its contractions. It
does not change the stage or assume that a dynamic attention operand is a
weight. A consumer must still bind the stage to the captured checkpoint
before replacing any live packing with stored MX bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_arg(value) -> tuple[int | None, tuple[str, ...]]:
    """Follow only layout changes and exact BF16-to-F32 element casts."""
    from xdsl.ir import BlockArgument

    path = []
    while not isinstance(value, BlockArgument):
        op = value.owner
        name = str(op.name)
        if name == "linalg.transpose":
            value = op.operands[0]
        elif name == "linalg.generic":
            body = list(op.body.blocks[0].ops)
            if (len(op.inputs) != 1 or len(body) != 2
                    or str(body[0].name) != "arith.extf"
                    or str(body[1].name) != "linalg.yield"
                    or body[0].operands[0] != op.body.blocks[0].args[0]
                    or body[1].operands[0] != body[0].results[0]):
                return None, tuple(path + ["dynamic:" + name])
            value = op.inputs[0]
        else:
            return None, tuple(path + ["dynamic:" + name])
        path.append(name)
    return value.index, tuple(path)


def build_inventory(build: Path, stage: Path) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_file

    build, stage = Path(build), Path(stage)
    # Merlin's target hook may place the routed module in its final host ABI
    # artifact while retaining the earlier prepared source at the root.
    final_host = build / "device_host_abi/mx_vpu_elementwise.mlir"
    source = final_host if final_host.is_file() else build / "model.prepared.mlir"
    signatures = json.loads((build / "device_signatures.json").read_text())
    catalog = json.loads((build / "device_catalog/mx_catalog.json").read_text())
    binding = json.loads((build / "device/catalog_binding.json").read_text())
    stage_receipt = json.loads((build / "mx_stage_probe.json").read_text())
    weights_path = stage / "weights.safetensors"
    manifest_path = stage / "weights.safetensors.manifest.json"
    weights = json.loads(manifest_path.read_text())
    if (signatures.get("model_sha256") != catalog.get("source_sha256")
            or not catalog.get("coverage_complete")
            or stage_receipt.get("capture_model_sha256") != _sha(stage / "model.mlir")
            or not weights_path.is_file()):
        raise ValueError("MX stage source or checkpoint binding is incomplete")
    module = parse_mlir_file(source)
    calls = [op for op in module.walk()
             if str(op.name) == "func.call"
             and str(op.callee).lstrip("@") in signatures["signatures"]]
    routed = signatures["routed"]
    bindings = {row["region"]: row for row in catalog["bindings"]}
    if (len(calls) > len(routed) or len(bindings) != len(routed)
            or len({row["source_region"] for row in routed}) != len(routed)
            or binding.get("source_sha256") != catalog["source_sha256"]):
        raise ValueError("MX routed call coverage is incomplete")
    kernel_for = binding["entry_to_kernel"]
    by_kernel = {row["symbol"]: row for row in catalog["kernels"]}
    target_by_kernel = {}
    for source_binding in bindings.values():
        kernel = source_binding["symbol"]
        dtype = source_binding["target_dtypes"]
        if kernel in target_by_kernel and target_by_kernel[kernel] != dtype:
            raise ValueError("one MX kernel has inconsistent target dtypes")
        target_by_kernel[kernel] = dtype
    route_by_symbol = {}
    for route in routed:
        route_by_symbol.setdefault(route["symbol"], []).append(route)
    exact_sequence = len(calls) == len(routed) and all(
        str(call.callee).lstrip("@") == route["symbol"]
        for call, route in zip(calls, routed, strict=True))
    rows = []
    for index, call in enumerate(calls):
        symbol = str(call.callee).lstrip("@")
        kernel = kernel_for.get(symbol)
        candidates = route_by_symbol.get(symbol, [])
        if kernel not in by_kernel or not candidates:
            raise ValueError("MX live call has no bound catalog kernel")
        route = (routed[index] if exact_sequence else
                 candidates[0] if len(candidates) == 1 else None)
        region = route["source_region"] if route else None
        if route:
            source_binding = bindings.get(region)
            if (source_binding is None or source_binding["symbol"] != kernel
                    or route["source_operation_ordinal"]
                    != source_binding["source_operation_ordinal"]
                    or route["tensor_types"] != source_binding["tensor_types"]):
                raise ValueError("MX live call source identity changed")
        arg, transform = _source_arg(call.operands[1])
        parameter = weights.get(str(arg)) if arg is not None else None
        shape = (list(call.operands[1].type.get_shape())
                 if hasattr(call.operands[1].type, "get_shape") else [])
        if arg is not None and (not isinstance(parameter, dict)
                                or parameter.get("kind") != "param"
                                or not isinstance(parameter.get("shape"), list)):
            raise ValueError(f"MX RHS source argument {arg} is not a captured parameter")
        rows.append({
            "region": region, "callee": symbol,
            "catalog_kernel": kernel,
            "target_dtypes": target_by_kernel[kernel],
            "rhs_shape": shape,
            "classification": ("checkpoint_parameter_candidate" if parameter
                               else "dynamic_rhs"),
            "parameter_arg": arg if parameter else None,
            "parameter_name": parameter["weight"] if parameter else None,
            "parameter_shape": parameter["shape"] if parameter else None,
            "parameter_dtype": parameter["dtype"] if parameter else None,
            "transforms_from_rhs": list(transform),
        })
    counts = Counter(row["classification"] for row in rows)
    return {
        "schema": "mx_gemmini.immutable_rhs_inventory.v1",
        "status": "provenance_audit_only",
        "stage_model_sha256": _sha(stage / "model.mlir"),
        "prepared_model_sha256": _sha(source),
        "routed_model_file": str(source.relative_to(build)),
        "catalog_source_sha256": catalog["source_sha256"],
        "weights_sha256": _sha(weights_path),
        "weights_manifest_sha256": _sha(manifest_path),
        "catalog_routed_sites": len(routed),
        "live_calls_after_preparation": len(calls),
        "exact_route_sequence": exact_sequence,
        "counts": dict(sorted(counts.items())),
        "sites": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = build_inventory(args.build, args.stage)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
