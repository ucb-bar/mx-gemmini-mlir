"""Project Merlin's target-neutral model inventory onto MX planning facts.

Merlin owns the complete linalg-on-tensors parser and graph/dataflow record.
This adapter only checks the MX candidate shapes and binds a selected target
profile. It does not make a placement or claim an executable model.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib

from .target_profile import profile_sha256
from .merlin_attention_pairs import find_attention_pairs


def build_mx_worklist_from_merlin(mlir_text: str, profile: dict) -> dict:
    """Read a model through Merlin, then produce the existing MX worklist schema.

    Merlin is optional for standalone MX kernel development. Full-model
    integration supplies it explicitly; a missing installation is an error,
    never a reason to silently use this package's older local parser.
    """
    try:
        from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
    except ImportError as error:
        raise RuntimeError("Merlin model inventory is required for whole-model MX planning") from error

    if 'prov.level = "linalg-on-tensors"' not in mlir_text:
        raise ValueError("Merlin inventory needs linalg-on-tensors provenance")
    inventory = parse_linalg_mlir(mlir_text)
    if inventory.get("level") != "linalg-on-tensors":
        raise ValueError("Merlin returned a different model inventory level")
    groups: OrderedDict[tuple[str, str], dict] = OrderedDict()
    matmuls: list[dict] = []
    batch_matmuls: list[dict] = []
    for op in inventory["ops"]:
        prov = op.get("prov") or {}
        if prov.get("family") == "contraction":
            region, kind = prov.get("region_id"), prov.get("op")
            if not region or not kind:
                raise ValueError("Merlin contraction lacks stable source provenance")
            group = groups.setdefault((region, kind), {"fqns": set(), "operations": []})
            if prov.get("fqn"):
                group["fqns"].add(prov["fqn"])
            group["operations"].append(op["kind"])
        if op["kind"] == "linalg.generic" and prov.get("family") == "contraction" and prov.get("op") == "batch_matmul":
            if not prov.get("region_id"):
                raise ValueError("Merlin batch matmul has no source region identity")
            if prov.get("aten") != "aten.bmm.default":
                continue  # retain this region below as a host-owned candidate
            if not op["reduction_dims"]:
                continue  # source provenance can survive legal reduction elimination
            inputs, outputs, results = op["ins"], op["outs"], op["results"]
            if len(inputs) != 2 or len(outputs) != 1 or len(results) != 1:
                continue
            types = [*inputs, *outputs, *results]
            if any(len(t["shape"]) != 3 or t["dtype"] not in {"f32", "bf16"} for t in types):
                continue
            if len({t["dtype"] for t in types}) != 1:
                continue
            (batch, m, k), (rhs_batch, rhs_k, n), output, result = [t["shape"] for t in types]
            if (not all(type(dim) is int and dim > 0 for dim in
                        (batch, m, k, rhs_batch, rhs_k, n)) or
                    batch != rhs_batch or k != rhs_k or
                    output != [batch, m, n] or result != output):
                continue
            batch_matmuls.append({
                "fqn": prov.get("fqn"), "region_id": prov["region_id"],
                "source_op": "batch_matmul", "shape_bmnk": [batch, m, n, k],
                "element_type": types[0]["dtype"],
                "status": "requires_explicit_mx_quantization_and_batched_lowering",
            })
        if op["kind"] != "linalg.matmul":
            continue
        if prov.get("family") != "contraction" or not prov.get("region_id") or not prov.get("fqn"):
            raise ValueError("Merlin matmul lacks contraction provenance")
        inputs, outputs, results = op["ins"], op["outs"], op["results"]
        if len(inputs) != 2 or len(outputs) != 1 or len(results) != 1:
            raise ValueError("Merlin matmul operand convention changed")
        types = [*inputs, *outputs, *results]
        if any(len(t["shape"]) != 2 or t["dtype"] not in {"f32", "bf16"} for t in types):
            raise ValueError("Merlin matmul needs static rank-two floating tensors")
        if len({t["dtype"] for t in types}) != 1:
            raise ValueError("Merlin matmul has mixed input or output element types")
        (m, k), (weight_k, n), output, result = [t["shape"] for t in types]
        if (not all(type(dim) is int and dim > 0 for dim in (m, k, weight_k, n)) or
                k != weight_k or output != [m, n] or result != output):
            raise ValueError("Merlin matmul has inconsistent static shapes")
        matmuls.append({
            "fqn": prov["fqn"], "region_id": prov["region_id"],
            "source_op": prov.get("op"), "shape_mnk": [m, n, k],
            "element_type": types[0]["dtype"],
            "k_requires_32_element_scale_padding": k % 32 != 0,
            "status": "requires_explicit_mx_quantization_and_program_lowering",
        })

    if any(len(group["fqns"]) > 1 for group in groups.values()):
        raise ValueError("Merlin contraction has conflicting module provenance")
    rank2_groups = {(site["region_id"], site["source_op"]) for site in matmuls}
    if len(rank2_groups) != len(matmuls):
        raise ValueError("Merlin contraction has ambiguous rank-two site identity")
    if len({site["region_id"] for site in batch_matmuls}) != len(batch_matmuls):
        raise ValueError("Merlin contraction has ambiguous batch matmul site identity")
    return {
        "schema": "mx_gemmini.model2mlir_contraction_worklist.v1",
        "status": "frontend_import_only_no_executable_placement",
        "source_mlir_sha256": hashlib.sha256(mlir_text.encode()).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "function": inventory["entry"],
        "legal_mx_compute_modes": profile["legal_compute"],
        "contraction_region_count": len(groups),
        "contraction_region_kinds": dict(sorted(Counter(kind for _, kind in groups).items())),
        "rank2_matmuls": matmuls,
        "rank3_batch_matmuls": batch_matmuls,
        "attention_pairs": find_attention_pairs(inventory),
        "other_contraction_regions": [
            {"fqn": next(iter(group["fqns"]), None), "region_id": region,
             "source_op": kind,
             "operations": dict(sorted(Counter(group["operations"]).items()))}
            for (region, kind), group in groups.items()
            if (region, kind) not in rank2_groups
        ],
    }
