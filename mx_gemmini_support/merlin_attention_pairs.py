"""Recognize QK -> softmax -> PV from Merlin's structural SSA inventory.

Module names are useful diagnostics but cannot establish attention dataflow:
functional models often omit them. Only a unique producer path is selected.
"""

from __future__ import annotations


def _parents(op: dict) -> list[int]:
    return [source[1] for value in op["ins"]
            if (source := value["source"])[0] == "op"]


def _nearest(ops: list[dict], start: int, predicate, *, skip_region: str | None = None) -> set[int]:
    """Find the first matching producers on every backward SSA path."""
    found: set[int] = set()
    seen: set[int] = set()
    stack = [start]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        op = ops[index]
        if predicate(op) and op["prov"].get("region_id") != skip_region:
            found.add(index)
            continue
        stack.extend(_parents(op))
    return found


def find_attention_pairs(inventory: dict) -> list[dict]:
    """Return unambiguous source QK/PV pairs with an intervening softmax.

    The rank-three source contraction is checked separately by the MX worklist
    and catalog binder. This analysis supplies the relationship, never an
    independent claim that an arbitrary contraction is MX legal.
    """
    ops = inventory["ops"]

    def batch(op: dict) -> bool:
        prov = op["prov"]
        return (prov.get("family") == "contraction" and
                prov.get("op") == "batch_matmul" and
                prov.get("aten") == "aten.bmm.default" and
                op["kind"] == "linalg.generic" and
                len(op["ins"]) == 2 and len(op["results"]) == 1)

    def softmax(op: dict) -> bool:
        prov = op["prov"]
        return (prov.get("family") == "normalization" and
                prov.get("op") == "softmax" and
                prov.get("aten") == "aten._softmax.default" and
                bool(prov.get("region_id")))

    candidates = {op["prov"]["region_id"]: op for op in ops
                  if batch(op) and op["prov"].get("region_id")}
    pairs: list[dict] = []
    used_qk: set[str] = set()
    for pv_region, pv in candidates.items():
        first = pv["ins"][0]["source"]
        if first[0] != "op":
            continue
        boundaries = _nearest(ops, first[1], softmax, skip_region=pv_region)
        if len({ops[index]["prov"].get("region_id") for index in boundaries}) != 1:
            continue
        qks = set()
        for boundary in boundaries:
            for parent in _parents(ops[boundary]):
                qks.update(_nearest(ops, parent, batch, skip_region=pv_region))
        qk_regions = {ops[index]["prov"].get("region_id") for index in qks}
        if len(qk_regions) != 1:
            continue
        qk_region = qk_regions.pop()
        if (qk_region not in candidates or qk_region in used_qk or
                candidates[qk_region]["results"][0]["shape"] !=
                pv["ins"][0]["shape"]):
            continue
        used_qk.add(qk_region)
        pairs.append({"qk_region_id": qk_region, "pv_region_id": pv_region,
                      "softmax_region_id": ops[next(iter(boundaries))]["prov"]["region_id"]})
    return pairs
