"""The portable frontend must preserve the two contraction SSA edges."""

from pathlib import Path

import pytest

from mx_gemmini_support.standard_chain_handoff import (
    portable_chain_manifest, portable_chain_shapes, validate_portable_chain)
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_portable_chains import _source_order_wrapper


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxGemminiRocketConfig.json")
GRAPH = '''builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<64x64xf32>, %b1: tensor<64x64xf32>, %b2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %e1 = tensor.empty() : tensor<64x64xf32>
    %z1 = arith.constant 0.000000e+00 : f32
    %f1 = linalg.fill ins(%z1 : f32) outs(%e1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %c1 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%a, %b1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%f1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %e2 = tensor.empty() : tensor<64x64xf32>
    %z2 = arith.constant 0.000000e+00 : f32
    %f2 = linalg.fill ins(%z2 : f32) outs(%e2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %c2 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%c1, %b2 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%f2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    func.return %c2 : tensor<64x64xf32>
  }
}
'''
SOURCE_SHA = "a" * 64
HEADER_SHA = "b" * 64


@pytest.mark.parametrize("precision", ("FP8", "FP4", "FP6"))
def test_two_site_capture_binds_legal_mx_mode(precision: str) -> None:
    manifest = portable_chain_manifest(
        GRAPH, PROFILE, precision=precision,
        source_driver_sha256=SOURCE_SHA, source_header_sha256=HEADER_SHA)
    assert manifest["sites"] == [
        {"site_id": site, "status": "quantized", "format": "mx" + precision.lower(),
         "shape": [64, 64, 64]}
        for site in ("functional:matmul", "functional:matmul_1")]
    assert validate_portable_chain(
        GRAPH, manifest, PROFILE, precision=precision,
        source_driver_sha256=SOURCE_SHA,
        source_header_sha256=HEADER_SHA) == ((64, 64, 64),) * 2
    with pytest.raises(ValueError, match="manifest differs"):
        validate_portable_chain(
            GRAPH, manifest, PROFILE, precision=precision,
            source_driver_sha256="c" * 64,
            source_header_sha256=HEADER_SHA)


@pytest.mark.parametrize("changed,reason", [
    (GRAPH.replace("ins(%c1, %b2", "ins(%a, %b2"), "SSA edge"),
    (GRAPH.replace("0.000000e+00", "1.000000e+00", 1), "initializer"),
    (GRAPH.replace('prov.region_id = "matmul_1"',
                   'prov.region_id = "matmul_0"'), "provenance"),
    (GRAPH.replace("func.return %c2", "func.return %c1"), "return"),
    (GRAPH.replace('prov.level = "linalg-on-tensors"',
                   'prov.level = "torch"'), "provenance"),
])
def test_portable_chain_rejects_changed_graph(changed: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        portable_chain_shapes(changed)


def test_portable_chain_rejects_unavailable_precision_and_source_binding() -> None:
    fp6_only = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                            "MxE3M2OnlyGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="legal|supported|compute"):
        portable_chain_manifest(
            GRAPH, fp6_only, precision="FP8",
            source_driver_sha256=SOURCE_SHA,
            source_header_sha256=HEADER_SHA)
    with pytest.raises(ValueError, match="pinned source hashes"):
        portable_chain_manifest(
            GRAPH, PROFILE, precision="FP8",
            source_driver_sha256=SOURCE_SHA,
            source_header_sha256="invalid")


def test_source_checker_calls_the_public_object_in_its_declared_abi_order() -> None:
    driver = "void mx_issue(const void *, const void *);\nvoid run(void) { mx_issue(a, b); }\n"
    reordered = _source_order_wrapper(driver, ("a", "b"), ["b", "a"])
    assert "mx_issue_source_order(a, b);" in reordered
    assert "mx_issue(b, a);" in reordered
    with pytest.raises(ValueError, match="ABI differ"):
        _source_order_wrapper(driver, ("a", "b"), ["a", "c"])
