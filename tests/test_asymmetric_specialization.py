"""The mixed precision path needs an explicit source recipe and legal RTL mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.asymmetric_specialization import (ASYM_CELL, emit_baremetal,
                                                           source_recipe, specialize_handoff)
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (ROOT / "docs/evidence/model2mlir_radiance_mx_gemm_20261006.mlir").read_text()
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _profile(name: str) -> dict:
    return load_profile(PROFILES / f"{name}.json")


def _recipe(profile: dict) -> dict:
    return {"schema": "mx_gemmini.asymmetric_source_recipe.v1",
            "site_id": "functional:matmul", "shape": [64, 64, 64],
            "frontend_capture_format": "mxfp8", "profile_sha256": profile_sha256(profile),
            "source_driver_sha256": "a" * 64, "source_header_sha256": "b" * 64,
            "compute": ASYM_CELL.copy()}


def test_explicit_recipe_selects_mixed_lut_mode():
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    bound = specialize_handoff(CAPTURE, profile, _recipe(profile))
    assert 'activation_projection = "lut"' in bound
    assert 'weight_format = "fp4_e2m1"' in bound
    assert 'pe_mode = 10 : i32' in bound
    assert verify_ir(bound, profile)["contracts"] == 1


def test_legacy_binder_cannot_silently_change_to_asymmetric():
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    with pytest.raises(ValueError, match="capture cannot change operand projection"):
        bind_handoff(CAPTURE, profile, {"functional:matmul": ASYM_CELL})


@pytest.mark.parametrize("change", [
    {"source_header_sha256": "invalid"},
    {"site_id": "functional:other"},
    {"compute": {**ASYM_CELL, "pe_mode": 6}},
])
def test_recipe_changes_are_rejected(change):
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    recipe = _recipe(profile) | change
    with pytest.raises(ValueError, match="recipe"):
        specialize_handoff(CAPTURE, profile, recipe)


def test_vpu_profile_rejects_mixed_compute_mode():
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    with pytest.raises(ValueError, match="absent"):
        specialize_handoff(CAPTURE, profile, _recipe(profile))


def test_profile_bound_capture_cannot_be_specialized_twice():
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    bound = specialize_handoff(CAPTURE, profile, _recipe(profile))
    with pytest.raises(ValueError, match="unbound frontend handoff"):
        specialize_handoff(bound, profile, _recipe(profile))


def test_emitter_rejects_a_legal_but_different_physical_mode(tmp_path):
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    source = tmp_path / "matmul_tiled_asym_e4m3_fp4_64x64.c"
    header = tmp_path / "matmul_data_asym_e4m3_fp4.h"
    source.write_text("\n".join((
        '#include "include/matmul_data_asym_e4m3_fp4.h"',
        "#define USE_LUT 1", "#define MX_ALTFMT 0",
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K")))
    header.write_text("\n".join((
        "#define MATMUL_M   64", "#define MATMUL_K   64", "#define MATMUL_N   64",
        "A_in_hw[32][64]", "B_in[MATMUL_K][MATMUL_N / 2]", "A_lut[32][4]",
        "A_scales_row[MATMUL_GK][MATMUL_M]",
        "B_scales_col[MATMUL_GK][MATMUL_N]",
        "C_out_bf16[MATMUL_M][MATMUL_N]")))
    recipe = source_recipe(source, header, profile)
    bound = specialize_handoff(CAPTURE, profile, recipe)
    assert "gemmini_mx_load_lut_dt" in emit_baremetal(
        bound, profile, recipe, source=source, header=header)
    altered = bound.replace('activation_projection = "lut"',
                            'activation_projection = "direct"').replace(
                                'pe_mode = 10 : i32', 'pe_mode = 6 : i32')
    assert verify_ir(altered, profile)["contracts"] == 1
    with pytest.raises(ValueError, match="differs from recipe"):
        emit_baremetal(altered, profile, recipe, source=source, header=header)
