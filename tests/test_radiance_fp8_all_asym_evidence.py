"""Validate the Radiance direct FP8 source run on Nicolas's DIM16 Spike."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp8_dim16_all_asym_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radiance_direct_fp8_mode8_reproduces_on_nicolas_spike() -> None:
    capture = json.loads((EVIDENCE / "capture_receipt.json").read_text())
    first = json.loads((EVIDENCE / "first.json").read_text())
    repro = json.loads((EVIDENCE / "repro.json").read_text())
    profile = json.loads((ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                          "MxAllAsymGemminiRocketConfig.json").read_text())
    compute = {
        "activation_format": "fp8_e4m3", "activation_projection": "direct",
        "weight_format": "fp8_e4m3", "weight_projection": "direct",
        "pe_mode": 8,
    }
    assert compute in profile["legal_compute"]
    bound_mlir = (EVIDENCE / "profile_bound.mlir").read_text()
    assert all(fragment in bound_mlir for fragment in (
        'activation_format = "fp8_e4m3"',
        'activation_projection = "direct"',
        'weight_format = "fp8_e4m3"',
        'weight_projection = "direct"',
        'pe_mode = 8 : i32',
    ))
    assert capture["source_revision"] == "d4732fb41c2d55088050c0399a4c6243b53e5fbd"
    assert capture["model2mlir_revision"] == "7485a829c0195af0ec42820837d609e62e466564"
    assert capture["source_data_header_origin"] == "generated_or_untracked"
    assert capture["source_shape"] == [64, 64, 64]
    assert capture["source_data_header_sha256"] == _sha(
        EVIDENCE / "generated_source_header.h")
    assert capture["source_mlir_sha256"] == _sha(EVIDENCE / "model2mlir.mlir")
    assert capture["handoff_mlir_sha256"] == _sha(EVIDENCE / "handoff.mlir")
    assert capture["target_binding"]["bound_mlir_sha256"] == _sha(
        EVIDENCE / "profile_bound.mlir")
    assert first["bound_mlir_sha256"] == _sha(EVIDENCE / "payload_bound.mlir")
    assert first["files_sha256"]["physical_program.json"] == _sha(
        EVIDENCE / "physical_program.json")
    assert first["files_sha256"]["mx_issue.c"] == _sha(EVIDENCE / "mx_issue.c")
    for manifest in (first, repro):
        assert manifest["status"] == "source_golden_matched_on_pinned_spike"
        assert manifest["compared_bf16_outputs"] == 4096
        assert manifest["spike_exit_code"] == 0
        assert manifest["compiler_revision"].startswith("b6019d0")
        assert manifest["rtl_revision"].startswith("266c593")
        assert manifest["profile_sha256"] == capture["target_binding"]["profile_sha256"]
        assert manifest["source_header_sha256"] == _sha(
            EVIDENCE / "generated_source_header.h")
    first["build_log_sha256"].pop("link.log")
    repro["build_log_sha256"].pop("link.log")
    assert first == repro
