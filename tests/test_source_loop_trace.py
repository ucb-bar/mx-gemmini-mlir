"""Check the source-bound MLIR loop trace against the current local kernels."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_loop_trace import (
    _software_geometry, _source_command_abi, trace_bound_source_gemm_loops)
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
BOUND = ROOT / "docs/evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir"
DRIVER = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp"


@pytest.mark.skipif(not DRIVER.is_file() or not (RTL / "software/gemmini-rocc-tests/include/gemmini.h").is_file(),
                    reason="requires current Radiance kernels and Nicolas Gemmini submodule")
def test_four_wave_bound_handoff_traces_source_loop_packets():
    profile = load_profile(PROFILE, rtl_root=RTL)
    result = trace_bound_source_gemm_loops(
        BOUND.read_text(), read_source_gemm(DRIVER), profile=profile,
        source_root=SOURCE, rtl_root=RTL)
    assert result["status"] == "symbolic_matrix_loop_trace_only"
    assert (result["source_layout_c_row"], result["target_layout_c_row"]) == (3072, 1024)
    assert result["wave_count"] == 4
    packets = result["packets"]
    assert len(packets) == 28  # 4 prefetches + 4 computes, each 3 commands, and 4 scale selects.
    assert [(p["phase"], p["wave"]) for p in packets if p["funct"] == 26] == [
        ("select_scales", i) for i in range(4)]
    assert [p["funct"] for p in packets[:3]] == [9, 24, 8]
    assert packets[1]["rs1"] == 0 and packets[1]["rs2"] == 16384
    computes = [p for p in packets if p["phase"] == "compute" and p["funct"] == 8]
    assert [p["rs1"] for p in computes] == [0, 1, 1, 1]
    assert [p["rs2"] for p in computes] == [
        (1024 << 32) | 0x2b8,
        (1024 << 32) | 0x2b8,
        (1024 << 32) | 0x2b8,
        (1024 << 32) | 0x238,
    ]


@pytest.mark.skipif(not DRIVER.is_file() or not (RTL / "software/gemmini-rocc-tests/include/gemmini.h").is_file(),
                    reason="requires current Radiance kernels and Nicolas Gemmini submodule")
def test_bound_handoff_must_match_source_precision():
    profile = load_profile(PROFILE, rtl_root=RTL)
    changed = (BOUND.read_text()
               .replace('activation_format = "fp8_e4m3"', 'activation_format = "fp4_e2m1"')
               .replace('weight_format = "fp8_e4m3"', 'weight_format = "fp4_e2m1"')
               .replace('pe_mode = 8 : i32', 'pe_mode = 0 : i32'))
    with pytest.raises(ValueError, match="activation mode differs"):
        trace_bound_source_gemm_loops(changed, read_source_gemm(DRIVER),
                                      profile=profile, source_root=SOURCE, rtl_root=RTL)


@pytest.mark.skipif(not (RTL / "software/gemmini-rocc-tests/include/gemmini.h").is_file() or
                    not (SOURCE / "lib/include/mxgemmini_mmio.h").is_file(),
                    reason="requires current source command headers")
def test_loop_trace_refuses_changed_source_macro(tmp_path):
    header = RTL / "software/gemmini-rocc-tests/include/gemmini.h"
    changed = header.read_text().replace("0x200U | (skips)", "0x000U | (skips)")
    assert changed != header.read_text()
    replacement = tmp_path / "gemmini.h"
    replacement.write_text(changed)
    with pytest.raises(ValueError, match="loop macro differs"):
        _source_command_abi(replacement, SOURCE / "lib/include/mxgemmini_mmio.h")


@pytest.mark.skipif(not (RTL / "software/gemmini-rocc-tests/include/gemmini_params.h").is_file(),
                    reason="requires selected Gemmini software geometry")
def test_loop_trace_refuses_mismatched_software_scratchpad(tmp_path):
    params = RTL / "software/gemmini-rocc-tests/include/gemmini_params.h"
    changed = params.read_text().replace("#define BANK_ROWS 4096", "#define BANK_ROWS 2048")
    assert changed != params.read_text()
    replacement = tmp_path / "gemmini_params.h"
    replacement.write_text(changed)
    with pytest.raises(ValueError, match="geometry differs"):
        _software_geometry(replacement, load_profile(PROFILE, rtl_root=RTL))
