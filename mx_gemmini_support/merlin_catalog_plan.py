"""Exact source-binding plan for Merlin's MX dense device catalog.

This module reads Merlin's structural candidates and the MX precision plan.
It records source proposals that cannot yet bind to the current f32 pointer
ABI. A separate builder may compile the bound subset as an experimental
catalog; full numerical and resource qualification remain separate gates.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from .merlin_inventory import build_mx_worklist_from_merlin
from .model_precision_plan import build_model_precision_plan
from .target_profile import profile_sha256
from .weight_provenance import source_parameter_arg


_TARGET_DTYPES = {
    "mxfp4": ["f4E2M1FN", "f4E2M1FN", "bf16"],
    "mxfp8": ["f8E4M3FN", "f8E4M3FN", "bf16"],
}


def plan_merlin_catalog(prepared: Path, profile: dict, *,
                        device: str = "mx_gemmini",
                        checkpoint_manifest: dict | None = None,
                        cache_immutable_weights: bool = False,
                        precision_overrides: dict[str, str] | None = None) -> dict:
    """Bind every selected direct matrix contraction to exact prepared MLIR.

    The caller must select the same OOT target package on MERLIN_TARGET_PATH.
    Source region and operation ordinal are both checked because a region
    label alone can survive an accidental graph rewrite that changes its ABI.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.system.offload import catalog_source_contractions

    prepared = Path(prepared)
    if cache_immutable_weights and not isinstance(checkpoint_manifest, dict):
        raise ValueError("MX checkpoint cache needs a captured parameter manifest")
    if prepared.is_symlink() or not prepared.is_file():
        raise ValueError("MX catalog plan needs a regular prepared MLIR file")
    raw = prepared.read_bytes()
    text = raw.decode("utf-8")
    worklist = build_mx_worklist_from_merlin(text, profile)
    proposal = (build_model_precision_plan(
        worklist, profile, precision_overrides=precision_overrides)
        if precision_overrides else build_model_precision_plan(worklist, profile))
    selected = {site["region_id"]: site for site in proposal["sites"]
                if site["selected_format"] in _TARGET_DTYPES}
    if len(selected) != sum(
            site["selected_format"] in _TARGET_DTYPES for site in proposal["sites"]):
        raise ValueError("MX precision proposal has duplicate source regions")
    module = parse_mlir_file(prepared)
    ordinal_by_op = {op: ordinal for ordinal, op in enumerate(module.walk())}
    candidates = catalog_source_contractions(module, device, ("f32", "f32", "f32"))
    by_region: dict[str, list] = {}
    for op, shape in candidates:
        region = getattr(op.attributes.get("prov.region_id"), "data", None)
        if region in selected:
            by_region.setdefault(region, []).append((op, shape))
    ambiguous = {region: len(by_region[region]) for region in selected
                 if len(by_region.get(region, ())) > 1}
    if ambiguous:
        raise ValueError(f"MX catalog source regions are ambiguous: {ambiguous}")
    unbound = [{
        "region": region,
        "source_op": site.get("source_op"),
        "element_type": site.get("element_type"),
        "selected_format": site["selected_format"],
        "reason": "no zero-initialized direct f32 catalog contraction at this source region",
    } for region, site in selected.items() if region not in by_region]
    kernels: dict[tuple, dict] = {}
    issuers: dict[tuple, str] = {}
    bindings = []
    for region, site in selected.items():
        if region not in by_region:
            continue
        op, shape = by_region[region][0]
        rank = len(shape.parallel)
        dimensions = tuple(shape.parallel) + tuple(shape.reduction)
        expected = (tuple(site["shape_bmnk"]) if rank == 3 and "shape_bmnk" in site
                    else tuple(site["shape_mnk"]) if rank == 2 and "shape_mnk" in site
                    else ())
        if (rank not in {2, 3} or dimensions != expected or
                len(op.operands) != 3 or len(op.results) != 1 or
                tuple(shape.dtypes) != ("f32", "f32", "f32")):
            raise ValueError(f"MX selected region {region} changed shape or source ABI")
        tensor_types = [str(op.operands[0].type), str(op.operands[1].type),
                        str(op.results[0].type)]
        if any("f32" not in value for value in tensor_types):
            raise ValueError(f"MX selected region {region} lost its f32 tensor ABI")
        base = (rank, dimensions, site["selected_format"])
        cached_arg = None
        cached_name = None
        if cache_immutable_weights and rank == 2:
            arg, _transforms, source_shape = source_parameter_arg(op.operands[1])
            parameter = checkpoint_manifest.get(str(arg)) if arg is not None else None
            if (isinstance(parameter, dict) and parameter.get("kind") == "param"
                    and parameter.get("shape") == list(source_shape or ())
                    and parameter.get("dtype") in {"float32", "bfloat16"}
                    and isinstance(parameter.get("weight"), str)
                    and parameter["weight"]):
                cached_arg = arg
                cached_name = parameter["weight"]
        key = base + ((region,) if cached_arg is not None else ())
        kernel = kernels.get(key)
        if kernel is None:
            issuer_symbol = issuers.setdefault(base, f"mx_issue_{len(issuers)}")
            kernel = {
                "symbol": f"mx_dense_kernel_{len(kernels)}",
                "issuer_symbol": issuer_symbol,
                "rank": rank, "shape": list(dimensions),
                "precision": site["selected_format"],
                "target_dtypes": _TARGET_DTYPES[site["selected_format"]],
                "cache_immutable_weight": cached_arg is not None,
            }
            if cached_arg is not None:
                kernel["checkpoint_parameter_arg"] = cached_arg
                kernel["checkpoint_parameter_name"] = cached_name
            kernels[key] = kernel
        bindings.append({
            "region": region, "symbol": kernel["symbol"],
            "source_operation_ordinal": ordinal_by_op[op],
            "tensor_types": tensor_types,
            "target_dtypes": kernel["target_dtypes"],
        })
    counts = Counter(kernel["precision"] for kernel in kernels.values())
    cached = sum(kernel["cache_immutable_weight"] for kernel in kernels.values())
    result = {
        "schema": "mx_gemmini.merlin_catalog_plan.v1",
        "status": ("no_direct_f32_catalog_candidates" if not bindings else
                   "exact_source_plan_partial_precision_coverage" if unbound else
                   "exact_source_plan_no_executable_catalog"),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "device": device,
        "structural_candidates": len(candidates),
        "selected_operations": len(bindings),
        "precision_proposals": len(selected),
        "unbound_precision_proposals": unbound,
        "unique_kernels": len(kernels),
        "unique_issuers": len(issuers),
        "cached_checkpoint_kernels": cached,
        "unique_kernel_precision_counts": dict(sorted(counts.items())),
        "kernels": list(kernels.values()),
        "bindings": bindings,
    }
    if precision_overrides:
        result["precision_overrides"] = dict(sorted(precision_overrides.items()))
    return result
