"""The mixed precision path needs an explicit source recipe and legal RTL mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.asymmetric_specialization import (ASYM_CELL, DIRECT_CELL,
                                                           E4M3_E4M3_CELL,
                                                           E5M2_E5M2_CELL,
                                                           bind_asymmetric_payload,
                                                           emit_baremetal, lower_asymmetric_physical,
                                                           source_recipe,
                                                           specialize_handoff)
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


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


def test_packed_lut_readout_is_typed_and_profile_bound():
    profile = _profile("MxE4M3LutGemminiRocketConfig")
    recipe = _recipe(profile) | {
        "compute": E4M3_E4M3_CELL.copy(),
        "source_layout": {"output_format": "fp8_e4m3", "output_projection": "lut"},
    }
    bound = specialize_handoff(CAPTURE, profile, recipe)
    assert '"mx_gemmini.readout_quantized"' in bound
    assert 'output_projection = "lut"' in bound
    assert 'tensor<32x64xi8>, tensor<64x2xi8>' in bound
    assert verify_ir(bound, profile)["contracts"] == 1
    with pytest.raises(ValueError, match="packed LUT readout format"):
        specialize_handoff(CAPTURE, profile, recipe | {
            "source_layout": {"output_format": "fp4_e2m1", "output_projection": "lut"}})


def test_e5m2_lut_index_readout_uses_semantic_output_format():
    profile = _profile("MxE5M2GemminiRocketConfig")
    recipe = _recipe(profile) | {
        "compute": E5M2_E5M2_CELL.copy(),
        "source_layout": {"output_format": "fp8_e5m2", "output_projection": "lut"},
    }
    bound = specialize_handoff(CAPTURE, profile, recipe)
    assert 'output_format = "fp8_e5m2"' in bound
    assert 'output_projection = "lut"' in bound
    assert verify_ir(bound, profile)["contracts"] == 1
    with pytest.raises(ValueError, match="absent from selected profile"):
        verify_ir(bound.replace('output_projection = "lut"',
                                'output_projection = "direct"'), profile)


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
        "A_in_hw[32][64]", "B_in[MATMUL_K][MATMUL_N / 2]",
        "A_lut[32][4]", "B_lut[32][4]", "C_lut[32][4]",
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


@pytest.mark.parametrize("profile_name", [
    "TestMxGemminiRocketConfig", "TestRequantizerLutMxGemminiRocketConfig",
])
def test_direct_fp4_source_needs_no_lut_width(profile_name, tmp_path):
    source = tmp_path / "matmul_tiled_fp4_64x64.c"
    header = tmp_path / "matmul_fp4_64x64.h"
    source.write_text("\n".join((
        '#include "include/matmul_fp4_64x64.h"',
        "gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, "
        "ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 3, 0)",
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K)",
    )))
    header.write_text("\n".join((
        "#define MATMUL_M   64", "#define MATMUL_K   64", "#define MATMUL_N   64",
        "A_in_hw[32][64]", "B_in[64][32]", "A_scales_row[2][64]",
        "B_scales_col[2][64]", "C_out_bf16[64][64]",
    )))
    recipe = source_recipe(source, header, _profile(profile_name))
    assert recipe["compute"] == {
        "activation_format": "fp4_e2m1", "activation_projection": "direct",
        "weight_format": "fp4_e2m1", "weight_projection": "direct", "pe_mode": 0,
    }
    assert recipe["source_layout"]["use_lut"] is False


@pytest.mark.parametrize("variant,activation_bytes,mode,lut_loads,profile_name,lut_words", [
    ("e4m3_fp4", 2048, 10, 3, "MxAsymE4M3Fp4GemminiRocketConfig", 4),
    ("e4m3s_fp4", 4096, 6, 0, "MxAsymE4M3Fp4GemminiRocketConfig", 0),
    ("fp6_fp4", 2048, 3, 3, "MxAsymFp6Fp4GemminiRocketConfig", 3),
    ("fp4_fp6", 2048, 1, 3, "MxAsymFp4Fp6GemminiRocketConfig", 3),
    ("e4m3_e2m3", 2048, 9, 3, "MxAsymE4M3E2M3GemminiRocketConfig", 4),
    ("e5m2_fp4", 2048, 3, 3, "MxAsymE5M2Fp4GemminiRocketConfig", 4),
    ("e4m3s_e3m2", 4096, 7, 1, "MxAsymE4M3E3M2GemminiRocketConfig", 4),
    ("fp4_e4m3s", 2048, 2, 0, "MxAsymFp4E4M3GemminiRocketConfig", 0),
    ("e2m3_e3m2", 2048, 10, 3, "MxAsymE2M3E3M2GemminiRocketConfig", 3),
    ("e3m2_e4m3s", 2048, 5, 1, "MxAsymE3M2E4M3GemminiRocketConfig", 4),
])
def test_asymmetric_site_lowers_to_shared_command_ir(
        tmp_path, variant, activation_bytes, mode, lut_loads, profile_name, lut_words):
    profile = _profile(profile_name)
    source = tmp_path / f"matmul_tiled_asym_{variant}_64x64.c"
    header = tmp_path / f"matmul_data_asym_{variant}.h"
    packed_activation = variant not in {"e4m3s_fp4", "e4m3s_e3m2"}
    direct_weight_bytes = variant in {"fp4_e4m3s", "e3m2_e4m3s"}
    source.write_text("\n".join((
        f'#include "include/{header.name}"',
        ("((uint64_t)(0) << 5)" if variant == "fp4_e4m3s" else
         f"#define USE_LUT {int(lut_loads != 0)}"),
        f"#define MX_ALTFMT {int(variant in {'e5m2_fp4', 'e2m3_e3m2'})}",
        *(["((uint64_t)(1) << 31)", "((uint64_t)(1) << 12)"]
          if variant == "e4m3_e2m3" else []),
        *(["((uint64_t)(1) << 31)"] if variant == "e2m3_e3m2" else []),
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K")))

    def array(type_name: str, name: str, dims: str, count: int) -> str:
        return f"static const {type_name} {name}{dims} = {{\n" + \
            ", ".join("0" for _ in range(count)) + "\n};\n"

    header_text = ("#define MATMUL_M   64\n#define MATMUL_K   64\n"
                   "#define MATMUL_N   64\n")
    header_text += array("uint8_t", "A_in_hw" if packed_activation else "A_in",
                         "[32][64]" if packed_activation else "[MATMUL_M][MATMUL_K]",
                         activation_bytes)
    header_text += array("uint8_t", "B_in",
                         "[MATMUL_K][MATMUL_N]" if direct_weight_bytes else
                         "[MATMUL_K][MATMUL_N / 2]",
                         4096 if direct_weight_bytes else 2048)
    if lut_loads:
        for name in (("B_lut",) if variant == "e4m3s_e3m2" else
                     ("A_lut",) if variant == "e3m2_e4m3s" else
                     ("A_lut", "B_lut", "C_lut")):
            header_text += array("uint32_t", name, f"[32][{lut_words}]", 32 * lut_words)
    header_text += array("uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]", 128)
    header_text += array("uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]", 128)
    header_text += array("uint16_t", "C_out_bf16", "[MATMUL_M][MATMUL_N]", 4096)
    header.write_text(header_text)
    if variant == "e2m3_e3m2":
        with pytest.raises(ValueError, match="no unique legal profile mode"):
            source_recipe(source, header, _profile("MxAsymE4M3E3M2GemminiRocketConfig"))
    recipe = source_recipe(source, header, profile)
    bound = specialize_handoff(CAPTURE, profile, recipe)
    with pytest.raises(ValueError, match="payload differs from typed MLIR"):
        lower_asymmetric_physical(bound, profile, recipe,
                                  source=source, header=header)
    bound = bind_asymmetric_payload(bound, profile, recipe,
                                    source=source, header=header)
    assert 'mx.payload_manifest_sha256 = "' in bound
    assert classify(bound, profile)[0] == "asymmetric_source"
    if variant == "e4m3s_fp4":
        with pytest.raises(ValueError, match="lacks its recipe"):
            classify(bound.replace("mx.asymmetric_recipe_sha256",
                                   "mx.hidden_recipe_sha256"), profile)
    with pytest.raises(ValueError, match="already payload-bound"):
        bind_asymmetric_payload(bound, profile, recipe,
                                source=source, header=header)
    program, resources, resource_manifest = lower_asymmetric_physical(
        bound, profile, recipe, source=source, header=header)
    assert program.plan["pe_mode"] == mode
    assert program.plan["weight_projection"] == recipe["compute"]["weight_projection"]
    config_ex = next(step.command.rs1.immediate for step in program.steps
                     if isinstance(step.command, Command) and step.command.funct == 0)
    assert bool(config_ex & (1 << 31)) == (variant in {"e4m3_e2m3", "e2m3_e3m2"})
    assert bool(config_ex & (1 << 6)) == (variant in {"e5m2_fp4", "e2m3_e3m2"})
    assert len(resources["activation"]) == activation_bytes
    assert len(resources["weight"]) == (4096 if direct_weight_bytes else 2048)
    assert len(resources["golden_bf16"]) == 8192
    assert {"activation", "weight", "activation_scales", "weight_scales"} <= set(
        resource_manifest["resources"])
    assert sum(isinstance(step.command, Command) and step.command.funct == 29
               for step in program.steps) == lut_loads
    assert sum(isinstance(step.command, Command) and step.command.funct == 30
               for step in program.steps) == int(not lut_loads)
    generated = write_standalone_sources(tmp_path / "lowered", program, resources)
    expected_commands = (63 + (program.plan["tiles_i"] - 2) * 4 +
                         (program.plan["tiles_j"] - 2) * 4 +
                         (lut_loads if lut_loads else 1) - 3)
    assert generated["command_count"] == expected_commands
    assert "void mx_issue(" in (tmp_path / "lowered/mx_issue.c").read_text()


@pytest.mark.parametrize("mesh_dim,expected_tiles,expected_b_end", [
    (8, (4, 4, 8), 16384),
    (32, (1, 1, 2), 8192),
])
def test_asymmetric_source_layout_uses_selected_mesh_geometry(
        tmp_path, mesh_dim, expected_tiles, expected_b_end):
    profile = _profile(f"MxDim{mesh_dim}AllAsymGemminiRocketConfig")
    source = tmp_path / f"matmul_tiled_asym_fp4_fp6_64x64_dim{mesh_dim}.c"
    header = tmp_path / f"matmul_data_asym_fp4_fp6_dim{mesh_dim}.h"
    source.write_text("\n".join((
        f'#include "include/{header.name}"', f"#define DIM {mesh_dim}",
        "#define USE_LUT 1", "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K")))

    def array(ctype: str, name: str, shape: str, count: int) -> str:
        return (f"static const {ctype} {name}{shape} = {{\n" +
                ",".join("0" for _ in range(count)) + "\n};")

    header.write_text("\n".join((
        "#define MATMUL_M   64", "#define MATMUL_K   64", "#define MATMUL_N   64",
        array("uint8_t", "A_in_hw", "[32][64]", 2048),
        array("uint8_t", "B_in", "[MATMUL_K][MATMUL_N / 2]", 2048),
        *(array("uint32_t", name, "[32][3]", 96)
          for name in ("A_lut", "B_lut", "C_lut")),
        array("uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]", 128),
        array("uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]", 128),
        array("uint16_t", "C_out_bf16", "[MATMUL_M][MATMUL_N]", 4096))))
    recipe = source_recipe(source, header, profile)
    bound = bind_asymmetric_payload(
        specialize_handoff(CAPTURE, profile, recipe), profile, recipe,
        source=source, header=header)
    program, resources, _ = lower_asymmetric_physical(
        bound, profile, recipe, source=source, header=header)
    plan = program.plan
    assert (plan["tiles_i"], plan["tiles_j"], plan["tiles_k"]) == expected_tiles
    assert plan["mesh_dim"] == mesh_dim
    assert plan["b_row"] + expected_tiles[1] * expected_tiles[2] * mesh_dim == expected_b_end
    assert plan["c_row"] + 8192 // mesh_dim <= plan["b_row"]
    transfers = [step.command for step in program.steps
                 if isinstance(step.command, Command) and step.command.funct in (2, 3)]
    assert transfers and all(command.rs2.immediate >> 48 == mesh_dim for command in transfers)
    write_standalone_sources(tmp_path / "lowered", program, resources)
