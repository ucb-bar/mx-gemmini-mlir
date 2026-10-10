"""Check the last FP8 source driver against latest model2MLIR and Spike."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp8_512_tk256_latest_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latest_model2mlir_fp8_tk256_spike_evidence(tmp_path):
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert index["schema"] == "mx_gemmini.radiance_fp8_512_tk256_latest_spike.v1"
    assert index["model2mlir_revision"] == (
        "7485a829c0195af0ec42820837d609e62e466564")
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    for filename, digest in index["files_sha256"].items():
        assert _sha(EVIDENCE / filename) == digest
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    capture = json.loads((EVIDENCE / "capture_receipt.json").read_text())
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    bound = (EVIDENCE / "bound.mlir").read_text()
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["spike_exit_code"] == 0
    assert receipt["compared_bf16_outputs"] == 16384
    assert receipt["source_driver_sha256"] == _sha(EVIDENCE / "source_driver.cpp")
    assert receipt["source_header_sha256"] == _sha(EVIDENCE / "source_header.h")
    assert receipt["bound_mlir_sha256"] == _sha(EVIDENCE / "bound.mlir")
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert "0 BF16 mismatches" in (EVIDENCE / "spike.log").read_text()
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
    assert verify_ir(bound, profile)["source_resources"] == 4
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.output_format == "bf16"
    assert program.shape == (128, 128, 512)
    regenerated = write_standalone_sources(tmp_path / "regenerated", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert regenerated["files_sha256"][name] == _sha(EVIDENCE / name)
