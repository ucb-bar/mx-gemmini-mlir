"""The mixed precision path needs an explicit source recipe and legal RTL mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.asymmetric_specialization import (ASYM_CELL, DIRECT_CELL,
                                                           emit_baremetal, lower_asymmetric_physical,
                                                           source_recipe,
                                                           specialize_handoff)
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.standalone import write_standalone_sources
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


def test_direct_e4m3_fp4_uses_full_activation_rows_without_lut(tmp_path):
    profile = _profile("MxAsymE4M3Fp4GemminiRocketConfig")
    source = tmp_path / "matmul_tiled_asym_e4m3s_fp4_64x64.c"
    header = tmp_path / "matmul_data_asym_e4m3s_fp4.h"
    source.write_text("\n".join((
        '#include "include/matmul_data_asym_e4m3s_fp4.h"',
        "#define USE_LUT 0", "#define MX_ALTFMT 0",
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K")))
    header.write_text("\n".join((
        "#define MATMUL_M   64", "#define MATMUL_K   64", "#define MATMUL_N   64",
        "A_in[MATMUL_M][MATMUL_K]", "B_in[MATMUL_K][MATMUL_N / 2]",
        "A_scales_row[MATMUL_GK][MATMUL_M]",
        "B_scales_col[MATMUL_GK][MATMUL_N]",
        "C_out_bf16[MATMUL_M][MATMUL_N]")))
    recipe = source_recipe(source, header, profile)
    assert recipe["compute"] == DIRECT_CELL
    bound = specialize_handoff(CAPTURE, profile, recipe)
    emitted = emit_baremetal(bound, profile, recipe, source=source, header=header)
    assert 'activation_projection = "direct"' in bound
    assert 'pe_mode = 6 : i32' in bound
    assert "const int tiles_i = MATMUL_M / 16" in emitted
    assert "&A_in[i * DIM][k * DIM]" in emitted
    assert "gemmini_mx_load_lut_dt" not in emitted


@pytest.mark.parametrize("variant,activation_bytes,mode,lut_loads,profile_name,lut_words", [
    ("e4m3_fp4", 2048, 10, 3, "MxAsymE4M3Fp4GemminiRocketConfig", 4),
    ("e4m3s_fp4", 4096, 6, 0, "MxAsymE4M3Fp4GemminiRocketConfig", 0),
    ("fp6_fp4", 2048, 3, 3, "MxAsymFp6Fp4GemminiRocketConfig", 3),
    ("fp4_fp6", 2048, 1, 3, "MxAsymFp4Fp6GemminiRocketConfig", 3),
])
def test_asymmetric_site_lowers_to_shared_command_ir(
        tmp_path, variant, activation_bytes, mode, lut_loads, profile_name, lut_words):
    profile = _profile(profile_name)
    source = tmp_path / f"matmul_tiled_asym_{variant}_64x64.c"
    header = tmp_path / f"matmul_data_asym_{variant}.h"
    source.write_text("\n".join((
        f'#include "include/{header.name}"',
        f"#define USE_LUT {int(lut_loads != 0)}", "#define MX_ALTFMT 0",
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K")))

    def array(type_name: str, name: str, dims: str, count: int) -> str:
        return f"static const {type_name} {name}{dims} = {{\n" + \
            ", ".join("0" for _ in range(count)) + "\n};\n"

    header_text = ("#define MATMUL_M   64\n#define MATMUL_K   64\n"
                   "#define MATMUL_N   64\n")
    header_text += array("uint8_t", "A_in_hw" if lut_loads else "A_in",
                         "[32][64]" if lut_loads else "[MATMUL_M][MATMUL_K]",
                         activation_bytes)
    header_text += array("uint8_t", "B_in", "[MATMUL_K][MATMUL_N / 2]", 2048)
    if lut_loads:
        for name in ("A_lut", "B_lut", "C_lut"):
            header_text += array("uint32_t", name, f"[32][{lut_words}]", 32 * lut_words)
    header_text += array("uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]", 128)
    header_text += array("uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]", 128)
    header_text += array("uint16_t", "C_out_bf16", "[MATMUL_M][MATMUL_N]", 4096)
    header.write_text(header_text)
    recipe = source_recipe(source, header, profile)
    bound = specialize_handoff(CAPTURE, profile, recipe)
    program, resources, resource_manifest = lower_asymmetric_physical(
        bound, profile, recipe, source=source, header=header)
    assert program.plan["pe_mode"] == mode
    assert len(resources["activation"]) == activation_bytes
    assert len(resources["golden_bf16"]) == 8192
    assert {"activation", "weight", "activation_scales", "weight_scales"} <= set(
        resource_manifest["resources_sha256"])
    assert sum(isinstance(step.command, Command) and step.command.funct == 29
               for step in program.steps) == lut_loads
    assert sum(isinstance(step.command, Command) and step.command.funct == 30
               for step in program.steps) == int(not lut_loads)
    generated = write_standalone_sources(tmp_path / "lowered", program, resources)
    assert generated["command_count"] == (63 if lut_loads else 69)
    assert "void mx_issue(" in (tmp_path / "lowered/mx_issue.c").read_text()
