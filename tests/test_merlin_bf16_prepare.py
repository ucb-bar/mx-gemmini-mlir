"""BF16 source matmul boundaries for the MX dense conversion catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_bf16_prepare import MXBF16CatalogPreparation
from mx_gemmini_support.merlin_catalog_plan import plan_merlin_catalog
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


SOURCE = '''builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<2x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<2x32xbf16> {
    %empty = tensor.empty() : tensor<2x32xbf16>
    %zero = arith.constant 0.000000e+00 : bf16
    %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<2x32xbf16>) -> tensor<2x32xbf16>
    %mm = linalg.matmul {prov.region_id = "matmul_0", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "bfloat16", prov.module = "model", prov.fqn = "model.proj"} ins(%a, %b : tensor<2x32xbf16>, tensor<32x32xbf16>) outs(%init : tensor<2x32xbf16>) -> tensor<2x32xbf16>
    return %mm : tensor<2x32xbf16>
  }
}
'''


def test_selected_bf16_source_binds_existing_f32_catalog(tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE, encoding="utf-8")
    selected = MXBF16CatalogPreparation(PROFILE)(source, tmp_path / "prepared")
    assert source.read_text(encoding="utf-8") == SOURCE
    receipt = json.loads((selected.parent / "mx_bf16_catalog.json").read_text())
    assert receipt["widened_operations"] == 1
    assert "mx.source_bf16_widened" in selected.read_text()
    plan = plan_merlin_catalog(selected, PROFILE)
    assert plan["selected_operations"] == 1
    assert plan["precision_proposals"] == 1
    assert not plan["unbound_precision_proposals"]
    assert plan["bindings"][0]["tensor_types"] == [
        "tensor<2x32xf32>", "tensor<32x32xf32>", "tensor<2x32xf32>"]


def test_nonzero_bf16_init_refuses_widening(tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE.replace("0.000000e+00", "1.000000e+00"), encoding="utf-8")
    with pytest.raises(ValueError, match="not exact zero-init sources"):
        MXBF16CatalogPreparation(PROFILE)(source, tmp_path / "prepared")
