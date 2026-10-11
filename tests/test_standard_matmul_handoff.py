"""The current portable frontend graph must be the graph actually bound to MX."""

from pathlib import Path

import pytest

from mx_gemmini_support.standard_matmul_handoff import bind_standard_matmul
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE3M2OnlyGemminiRocketConfig.json")
GRAPH = """builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %fill = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%zero : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %result = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    func.return %result : tensor<128x128xf32>
  }
}
"""
MANIFEST = {"schema": "mx_gemmini.source_payload.v1", "site_id": "functional:matmul",
            "precision": "FP6", "shape_mnk": [128, 128, 128],
            "profile_sha256": profile_sha256(PROFILE)}


def test_portable_matmul_binds_one_legal_mx_site() -> None:
    bound, receipt = bind_standard_matmul(GRAPH, PROFILE, MANIFEST)
    assert verify_ir(bound, PROFILE)["contracts"] == 1
    assert receipt["shape_mnk"] == [128, 128, 128]
    assert receipt["mode"]["pe_mode"] == 4
    assert 'prov.quantization = "explicit:mx_gemmini"' in bound
    assert receipt == bind_standard_matmul(GRAPH, PROFILE, MANIFEST)[1]


@pytest.mark.parametrize("changed,reason", [
    (GRAPH.replace("0.000000e+00", "1.000000e+00"), "initializer"),
    (GRAPH.replace("%a, %b :", "%b, %a :"), "data flow"),
    (GRAPH.replace('prov.aten = "aten.mm.default"',
                   'prov.aten = "aten.add.Tensor"'), "provenance"),
    (GRAPH.replace("func.return %result", "func.return %fill"), "data flow"),
    (GRAPH.replace('prov.level = "linalg-on-tensors"', 'prov.level = "torch"'),
     "provenance"),
])
def test_portable_matmul_rejects_changed_semantics(changed: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        bind_standard_matmul(changed, PROFILE, MANIFEST)


def test_portable_matmul_rejects_payload_shape_and_precision_drift() -> None:
    with pytest.raises(ValueError, match="differs from the explicit MX source payload"):
        bind_standard_matmul(GRAPH, PROFILE, {**MANIFEST, "shape_mnk": [64, 128, 128]})
    with pytest.raises(ValueError, match="legal MX precision"):
        bind_standard_matmul(GRAPH, PROFILE, {**MANIFEST, "precision": "FP4"})
