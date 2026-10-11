"""Account for source softmax boundaries against the selected MX VPU profile.

This is a placement gate, not a VPU lowering. A directly legal BF16 rank-two
softmax must not silently fall back to the host while the stage builder lacks
that lowering. Converted f32 or higher-rank regions remain explicitly owned
by the host until a numerical conversion policy and schedule are implemented.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from math import prod

from .merlin_profile_contract import contract_for_profile
from .target_profile import profile_sha256


def audit_softmax_boundaries(mlir_text: str, profile: dict) -> dict:
    """Classify actual model2MLIR softmax input/output tensors by source region."""
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    inventory = parse_linalg_mlir(mlir_text)
    units = contract_for_profile(profile)["compute_units"]
    direct = any(unit["name"] == "vpu" and any(
        cap["family"] == "softmax" and "bf16" in cap["dtypes"] and 2 in cap["ranks"]
        for cap in unit["semantic_capabilities"]) for unit in units)
    regions: OrderedDict[str, dict] = OrderedDict()
    for op in inventory["ops"]:
        prov = op.get("prov") or {}
        if prov.get("op") != "softmax":
            continue
        region = prov.get("region_id")
        if (not region or prov.get("family") != "normalization" or
                prov.get("aten") != "aten._softmax.default"):
            raise ValueError("captured softmax lacks stable normalization provenance")
        row = regions.setdefault(region, {"region": region, "fqn": prov.get("fqn"),
                                          "generics": []})
        if row["fqn"] != prov.get("fqn"):
            raise ValueError(f"softmax region {region} has conflicting module provenance")
        if op["kind"] == "linalg.generic":
            if not op.get("ins") or len(op.get("results") or ()) != 1:
                raise ValueError(f"softmax region {region} lost its tensor result")
            row["generics"].append(op)
    records = []
    for region, row in regions.items():
        generics = row["generics"]
        if not generics:
            raise ValueError(f"softmax region {region} has no elementwise boundary")
        source = generics[0]["ins"][0]
        result = generics[-1]["results"][0]
        in_shape, out_shape = source["shape"], result["shape"]
        if (in_shape != out_shape or not in_shape or
                any(type(dim) is not int or dim <= 0 for dim in in_shape)):
            raise ValueError(f"softmax region {region} changed its static tensor shape")
        is_direct = direct and source["dtype"] == result["dtype"] == "bf16" and len(in_shape) == 2
        records.append({
            "region": region, "fqn": row["fqn"],
            "input_shape": in_shape, "output_shape": out_shape,
            "input_dtype": source["dtype"], "output_dtype": result["dtype"],
            "logical_rows": prod(in_shape[:-1]), "row_width": in_shape[-1],
            "profile_direct_match": is_direct,
            "owner": "unlowered_direct_vpu" if is_direct else "compiled_host",
            "reason": ("direct_bf16_rank2_vpu_lowering_required" if is_direct else
                       "profile_has_no_direct_vpu_softmax" if not direct else
                       "f32_or_higher_rank_requires_conversion_tiling_and_numeric_policy"),
        })
    return {
        "schema": "mx_gemmini.merlin_vpu_softmax_inventory.v1",
        "source_sha256": hashlib.sha256(mlir_text.encode()).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "softmax_regions": len(records),
        "direct_profile_matches_unlowered": sum(row["profile_direct_match"] for row in records),
        "host_owned_regions": sum(row["owner"] == "compiled_host" for row in records),
        "regions": records,
    }
