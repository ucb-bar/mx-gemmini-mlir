"""Check the source-derived FP4 capture and Spike evidence closure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_baremetal import emit_source_fp4_baremetal
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile


EVIDENCE = Path(__file__).resolve().parents[1] / "docs/evidence"
ROOT = EVIDENCE.parents[1]
SOURCE = Path(os.environ.get("RADIANCE_FP4_GENERATED_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
DRIVER = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_fp4_capture_and_execution_are_digest_bound():
    capture = json.loads((EVIDENCE / "model2mlir_radiance_mx_fp4_receipt_20261009.json").read_text())
    execution = json.loads((EVIDENCE / "source_fp4_64x64x128_spike_20261009.json").read_text())
    manifest = json.loads((EVIDENCE / "model2mlir_radiance_mx_fp4_manifest_20261009.json").read_text())
    trace = json.loads((EVIDENCE / "model2mlir_radiance_mx_fp4_loop_trace_20261009.json").read_text())
    paths = {
        "source_mlir_sha256": "model2mlir_radiance_mx_fp4_source_20261009.mlir",
        "handoff_mlir_sha256": "model2mlir_radiance_mx_fp4_handoff_20261009.mlir",
        "quantization_manifest_sha256": "model2mlir_radiance_mx_fp4_manifest_20261009.json",
    }
    for field, name in paths.items():
        assert capture[field] == _sha(EVIDENCE / name)
    bound = EVIDENCE / "model2mlir_radiance_mx_fp4_bound_20261009.mlir"
    header = EVIDENCE / "mxgemm.data.fp4.m64n64k128_20261009.h"
    c_file = EVIDENCE / "source_fp4_64x64x128_20261009.c"
    assert capture["target_binding"]["bound_mlir_sha256"] == _sha(bound)
    assert execution["bound_mlir_sha256"] == _sha(bound)
    assert capture["source_data_header_sha256"] == execution["source_data_header_sha256"] == _sha(header)
    assert execution["generated_c_sha256"] == _sha(c_file)
    assert capture["source_shape"] == [64, 64, 128]
    assert capture["source_tile"] == [64, 64, 64]
    assert capture["source_data_header_origin"] == "generated_or_untracked"
    assert capture["selected_site"]["format"] == "mxfp4"
    assert manifest["sites"][0]["format"] == "mxfp4"
    assert trace["wave_count"] == 2 and len(trace["scale_dma_packets"]) == 4
    assert execution["status"] == "source_golden_matched_on_pinned_spike"
    assert execution["spike_exit_code"] == 0
    assert execution["compared_bf16_outputs"] == 4096
    assert "0 BF16 mismatches" in execution["spike_output"]


@pytest.mark.skipif(not DRIVER.is_file() or
                    not DRIVER.with_name("mxgemm.data.fp4.m64n64k128.h").is_file() or
                    not (RTL / "software/libgemmini/gemmini.cc").is_file(),
                    reason="requires generated FP4 Radiance worktree and Nicolas RTL")
def test_fp4_emitter_recreates_saved_source_program():
    profile = load_profile(PROFILE, rtl_root=RTL)
    source = emit_source_fp4_baremetal(
        (EVIDENCE / "model2mlir_radiance_mx_fp4_bound_20261009.mlir").read_text(),
        read_source_gemm(DRIVER), profile=profile, source_root=SOURCE, rtl_root=RTL)
    assert source == (EVIDENCE / "source_fp4_64x64x128_20261009.c").read_text()
