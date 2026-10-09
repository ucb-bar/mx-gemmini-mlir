"""Check the typed FP6 source binding and pinned Spike evidence closure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_baremetal import emit_source_fp6_baremetal
from mx_gemmini_support.source_fp6 import read_source_fp6_payload
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
SOURCE = Path(os.environ.get("RADIANCE_FP6_SOURCE_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
DRIVER = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json"
BOUND = EVIDENCE / "model2mlir_radiance_mx_fp6_bound_20261009.mlir"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fp6_capture_payload_and_execution_receipts_agree():
    capture = json.loads((EVIDENCE / "model2mlir_radiance_mx_fp6_receipt_20261009.json").read_text())
    payload = json.loads((EVIDENCE / "source_fp6_payload_20261009.json").read_text())
    execution = json.loads((EVIDENCE / "source_fp6_128x128x2048_spike_20261009.json").read_text())
    trace = json.loads((EVIDENCE / "model2mlir_radiance_mx_fp6_loop_trace_20261009.json").read_text())
    assert capture["source_mlir_sha256"] == _sha(EVIDENCE / "model2mlir_radiance_mx_fp6_source_20261009.mlir")
    assert capture["handoff_mlir_sha256"] == _sha(EVIDENCE / "model2mlir_radiance_mx_fp6_handoff_20261009.mlir")
    assert capture["quantization_manifest_sha256"] == _sha(EVIDENCE / "model2mlir_radiance_mx_fp6_manifest_20261009.json")
    assert capture["selected_site"]["format"] == "mxfp6"
    assert capture["target_binding"]["bound_mlir_sha256"] == payload["bound_mlir_sha256"] == execution["bound_mlir_sha256"] == _sha(BOUND)
    assert capture["source_data_header_sha256"] == payload["source_header_sha256"] == execution["source_data_header_sha256"]
    assert capture["target_binding"]["profile_sha256"] == payload["target_profile_sha256"] == execution["target_profile_sha256"]
    assert payload["structural_policy_sha256"] == _sha(ROOT / "examples/fp6-source-line0-policy.yaml")
    assert payload["k_waves"] == trace["wave_count"] == 16
    assert len(trace["scale_dma_packets"]) == 32
    assert payload["lut_lines_per_bank"] == 64
    assert payload["payload_lengths"]["golden_bf16_bytes"] == 32768
    assert execution["generated_c_sha256"] == _sha(EVIDENCE / "source_fp6_128x128x2048_20261009.c")
    assert execution["status"] == "source_golden_matched_on_pinned_spike"
    assert execution["compared_bf16_outputs"] == 16384
    assert execution["spike_lut_scale_selector_workaround"] is True
    assert "0 BF16 mismatches" in execution["spike_output"]
    failed = json.loads((EVIDENCE / "source_fp6_alternating_scales_failed_20261009.json").read_text())
    assert failed["generated_c_sha256"] == _sha(EVIDENCE / "source_fp6_alternating_scales_failed_20261009.c")
    assert failed["source_data_header_sha256"] == execution["source_data_header_sha256"]
    assert failed["target_profile_sha256"] == execution["target_profile_sha256"]
    assert failed["status"] == "source_golden_failed_on_pinned_spike"
    assert "16368 BF16 mismatches" in failed["spike_output"]


@pytest.mark.skipif(not DRIVER.is_file() or
                    not DRIVER.with_name("mxgemm.data.fp6.m128n128k2048.h").is_file() or
                    not (RTL / "software/libgemmini/gemmini.cc").is_file(),
                    reason="requires the checked-in FP6 source and Nicolas RTL")
def test_fp6_source_recreates_saved_payload_and_program():
    kernel = read_source_gemm(DRIVER)
    payload = read_source_fp6_payload(kernel)
    receipt = json.loads((EVIDENCE / "source_fp6_payload_20261009.json").read_text())
    assert payload.header_sha256 == receipt["source_header_sha256"]
    assert payload.digests() == receipt["payload_digests"]
    profile = load_profile(PROFILE, rtl_root=RTL)
    generated = emit_source_fp6_baremetal(
        BOUND.read_text(), kernel, profile=profile, source_root=SOURCE, rtl_root=RTL)
    assert generated == (EVIDENCE / "source_fp6_128x128x2048_20261009.c").read_text()
