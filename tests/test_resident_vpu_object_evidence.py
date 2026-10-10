"""Audit the published MX+VPU object and full-output Spike replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_resident_vpu_object_6c9ed40"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_vpu_object_matches_source_and_fresh_replay():
    index = json.loads((EVIDENCE / "index.json").read_text())
    object_manifest = json.loads((EVIDENCE / "object_manifest.json").read_text())
    qualification = json.loads((EVIDENCE / "qualification_manifest.json").read_text())
    fresh_object = json.loads((EVIDENCE / "fresh_object_manifest.json").read_text())
    fresh_run = json.loads((EVIDENCE / "fresh_qualification_manifest.json").read_text())
    old = json.loads((ROOT / "docs/evidence/compiled_nicolas_connected_chain_266c593.json").read_text())
    capture = json.loads((ROOT / "docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/capture_receipt.json").read_text())
    assert index["status"] == "fresh_checkout_spike_full_output_match"
    assert all(_sha(EVIDENCE / name) == digest
               for name, digest in index["files_sha256"].items())
    assert object_manifest["compiler_revision"] == index["compiler_revision"]
    assert index["compiler_revision"].startswith("6c9ed40")
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert object_manifest["profile_sha256"] == index["profile_sha256"]
    assert object_manifest["bound_mlir_sha256"] == index["bound_mlir_sha256"]
    assert object_manifest["object_sha256"] == _sha(EVIDENCE / "mx_issue.o")
    assert object_manifest["issuer_c_sha256"] == _sha(EVIDENCE / "mx_issue.c")
    assert object_manifest["physical_program_sha256"] == _sha(
        EVIDENCE / "physical_program.json")
    assert [entry["slot"] for entry in object_manifest["buffer_abi"]] == sorted(
        entry["slot"] for entry in object_manifest["buffer_abi"])
    assert len(object_manifest["buffer_abi"]) == 11
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert object_manifest["embedded_golden_bytes"] == 0
    assert object_manifest["undefined_symbols"] == []
    assert object_manifest["object_sha256"] == old["object_sha256"]["mx_0.o"]
    assert qualification["issuer_origin"] == fresh_run["issuer_origin"] == (
        "linkable_resident_vpu_object")
    assert qualification["status"] == fresh_run["status"] == (
        "source_connected_full_chain_matched_on_pinned_spike")
    assert qualification["spike_exit_code"] == fresh_run["spike_exit_code"] == 0
    assert qualification["compared_bf16_values"] == 4096
    assert qualification["compared_fp8_codes"] == 8192
    assert qualification["compared_e8m0_scales"] == 256
    assert qualification["spike_log_sha256"] == old["spike_log_sha256"]
    assert qualification["spike_log_sha256"] == _sha(EVIDENCE / "spike.log")
    assert qualification["elf_sha256"] == _sha(EVIDENCE / "mx_program.elf")
    assert object_manifest["object_sha256"] == fresh_object["object_sha256"]
    assert qualification["elf_sha256"] == fresh_run["elf_sha256"]
    assert qualification["spike_log_sha256"] == fresh_run["spike_log_sha256"]
    assert "C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales mismatches" in (
        EVIDENCE / "spike.log").read_text()
