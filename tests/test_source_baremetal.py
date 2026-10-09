"""The source-bound FP8 program uses selected target geometry and source data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_baremetal import emit_source_fp8_baremetal
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
DRIVER = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp"
BOUND = ROOT / "docs/evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
EVIDENCE = ROOT / "docs/evidence/source_fp8_128x128x512_20261009.c"
RECEIPT = ROOT / "docs/evidence/source_fp8_128x128x512_spike_20261009.json"


@pytest.mark.skipif(not DRIVER.is_file() or not (RTL / "software/libgemmini/gemmini.cc").is_file(),
                    reason="requires current Radiance kernel and Gemmini source checkouts")
def test_emitted_source_program_matches_saved_execution_artifact():
    profile = load_profile(PROFILE, rtl_root=RTL)
    generated = emit_source_fp8_baremetal(
        BOUND.read_text(), read_source_gemm(DRIVER), profile=profile,
        source_root=SOURCE, rtl_root=RTL)
    receipt = json.loads(RECEIPT.read_text())
    assert generated == EVIDENCE.read_text()
    assert hashlib.sha256(generated.encode()).hexdigest() == receipt["generated_c_sha256"]
    assert "const uint32_t c_row = 1024;" in generated
    assert "const uint32_t scale_dest = odd * 4096;" in generated
    assert "gemmini_mx_load_scales_2d(&A_scales_row[group][0]" in generated
    assert "gemmini_mx_load_scales_2d(&B_scales_col[group][0]" in generated
    assert "gemmini_extended_mvin(&A_in[i * DIM][k_start + k * DIM]" in generated
    assert "gemmini_extended_mvin(&B_in[k_start + k * DIM][j * DIM]" in generated
    assert "C_hw[i][j] != C_out_bf16[i][j]" in generated
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["compared_bf16_outputs"] == 16384


@pytest.mark.skipif(not DRIVER.is_file() or not (RTL / "software/libgemmini/gemmini.cc").is_file(),
                    reason="requires current Radiance kernel and Gemmini source checkouts")
def test_bounded_program_rejects_unrelated_handoff():
    profile = load_profile(PROFILE, rtl_root=RTL)
    changed = BOUND.read_text().replace('weight_format = "fp8_e4m3"',
                                        'weight_format = "fp4_e2m1"')
    with pytest.raises(ValueError):
        emit_source_fp8_baremetal(changed, read_source_gemm(DRIVER), profile=profile,
                                 source_root=SOURCE, rtl_root=RTL)
