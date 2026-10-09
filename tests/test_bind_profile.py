"""Bind an actual model2MLIR/PyTorch capture, preserving its MX quantization."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (ROOT / "docs/evidence/model2mlir_radiance_mx_gemm_20261006.mlir").read_text()
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _profile(name):
    return load_profile(PROFILE_DIR / f"{name}.json")


def test_captured_pytorch_gemm_binds_to_standalone_mx():
    profile = _profile("MxGemminiRocketConfig")
    bound = bind_handoff(CAPTURE, profile)
    assert 'activation_format = "fp8_e4m3"' in bound
    assert 'weight_format = "fp8_e4m3"' in bound
    assert 'pe_mode = 8 : i32' in bound
    assert verify_ir(bound, profile)["contracts"] == 1


def test_legacy_capture_keeps_direct_projection_on_all_asym_build():
    profile = _profile("MxAllAsymGemminiRocketConfig")
    bound = bind_handoff(CAPTURE, profile)
    assert 'activation_projection = "direct"' in bound
    assert 'weight_projection = "direct"' in bound
    assert verify_ir(bound, profile)["contracts"] == 1
    choice = next(cell for cell in profile["legal_compute"] if
                  cell["activation_format"] == cell["weight_format"] == "fp8_e4m3" and
                  cell["activation_projection"] == cell["weight_projection"] == "lut")
    with pytest.raises(ValueError, match="capture cannot change operand projection"):
        bind_handoff(CAPTURE, profile, {"functional:matmul": choice})


def test_binder_refuses_to_change_frontend_quantization():
    profile = _profile("MxAllAsymGemminiRocketConfig")
    choice = next(cell for cell in profile["legal_compute"] if
                  cell["activation_format"] == cell["weight_format"] == "fp4_e2m1")
    with pytest.raises(ValueError, match="capture cannot change operand projection"):
        bind_handoff(CAPTURE, profile, {"functional:matmul": choice})
