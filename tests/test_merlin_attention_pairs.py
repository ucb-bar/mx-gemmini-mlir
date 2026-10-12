"""SSA attention pairing must survive absent FQNs without guessing by shape."""

from __future__ import annotations

from mx_gemmini_support.merlin_attention_pairs import find_attention_pairs


def _op(index: int, kind: str, region: str, family: str, aten: str,
        parents: tuple[int, ...] = (), shape: tuple[int, ...] = ()) -> dict:
    return {"id": index, "kind": kind,
            "prov": {"region_id": region, "family": family,
                     "op": "batch_matmul" if family == "contraction" else
                     "softmax" if family == "normalization" else "view",
                     "aten": aten},
            "ins": [{"source": ("op", parent), "shape": list(shape)}
                    for parent in parents],
            "results": [{"shape": list(shape)}]}


def test_functional_attention_pairs_by_ssa_without_module_provenance() -> None:
    shape = (15, 50, 163)
    qk = _op(0, "linalg.generic", "qk", "contraction", "aten.bmm.default",
             shape=shape)
    qk["ins"] = [{"source": ("arg", 0)}, {"source": ("arg", 1)}]
    reshape = _op(1, "tensor.collapse_shape", "view_0", "layout", "", (0,), shape)
    softmax = _op(2, "linalg.generic", "softmax_0", "normalization",
                  "aten._softmax.default", (1,), shape)
    cast = _op(3, "linalg.generic", "cast_0", "elementwise", "", (2,), shape)
    pv = _op(4, "linalg.generic", "pv", "contraction", "aten.bmm.default",
             (3,), shape)
    pv["ins"].append({"source": ("arg", 2)})
    unrelated = _op(5, "linalg.generic", "rotary", "contraction",
                    "aten.bmm.default", shape=shape)
    unrelated["ins"] = [{"source": ("arg", 3)}, {"source": ("arg", 4)}]
    assert find_attention_pairs({"ops": [qk, reshape, softmax, cast, pv, unrelated]}) == [
        {"qk_region_id": "qk", "pv_region_id": "pv",
         "softmax_region_id": "softmax_0"}]


def test_ambiguous_softmax_producers_are_not_selected() -> None:
    shape = (15, 50, 163)
    qk1 = _op(0, "linalg.generic", "qk1", "contraction", "aten.bmm.default", shape=shape)
    qk2 = _op(1, "linalg.generic", "qk2", "contraction", "aten.bmm.default", shape=shape)
    for op in (qk1, qk2):
        op["ins"] = [{"source": ("arg", 0)}, {"source": ("arg", 1)}]
    softmax = _op(2, "linalg.generic", "softmax_0", "normalization",
                  "aten._softmax.default", (0, 1), shape)
    pv = _op(3, "linalg.generic", "pv", "contraction", "aten.bmm.default", (2,), shape)
    pv["ins"].append({"source": ("arg", 2)})
    assert find_attention_pairs({"ops": [qk1, qk2, softmax, pv]}) == []
