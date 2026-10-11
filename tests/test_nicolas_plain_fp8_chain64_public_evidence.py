"""Audit the archived public-object replay of Nicolas's 64³ resident chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_plain_fp8_chain64_public_52e6132_266c593"
SOURCE_SHA = "54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf"
COMPILER_REV = "52e6132"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: str) -> dict:
    return json.loads((EVIDENCE / path).read_text())


def test_nicolas_plain_fp8_chain64_public_object_matches_source() -> None:
    capture = _json("capture/receipt.json")
    reference = _json("reference/build/artifact_manifest.json")
    obj = _json("object/object_manifest.json")
    parity = _json("parity/qualification_manifest.json")
    assert capture["source_sha256"] == reference["source_sha256"] == SOURCE_SHA
    assert (capture["compiler_revision"] == reference["compiler_revision"] ==
            obj["compiler_revision"] == parity["compiler_revision"])
    assert capture["compiler_revision"].startswith(COMPILER_REV)
    assert capture["model2mlir_revision"] == (
        "e9ded36eb85abf2d9097ac4dc11457c825853388")
    assert capture["opaque_calls"] == {}
    assert [site["shape"] for site in capture["sites"]] == [[64, 64, 64]] * 2
    assert reference["source_baseline"]["source_spike_exit_code"] == 0
    assert reference["spike_exit_code"] == parity["spike_exit_code"] == 0
    assert reference["status"] == "source_connected_chain_matched_on_pinned_spike"
    assert parity["status"] == "source_connected_object_matched_on_pinned_spike"
    assert obj["shape_mnk"] == parity["shape_mnk"] == [64, 64, 64]
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["undefined_symbols"] == []
    assert obj["object_sha256"] == parity["object_sha256"] == _sha(
        EVIDENCE / "object/mx_issue.o")
    assert obj["issuer_c_sha256"] == parity["source_issuer_sha256"] == _sha(
        EVIDENCE / "object/mx_issue.c")
    assert obj["object_sha256"] == reference["object_sha256"]["mx_0.o"]
    assert obj["issuer_c_sha256"] == reference["files_sha256"]["mx_issue.c"]
    assert parity["linked_elf_sha256"] == _sha(EVIDENCE / "parity/linked.elf")
    assert parity["spike_log_sha256"] == _sha(EVIDENCE / "parity/spike.log")
    for field, count in (("compared_c1_fp8_codes", 4096),
                         ("compared_c1_e8m0_scales", 128),
                         ("compared_c2_fp8_codes", 4096),
                         ("compared_c2_e8m0_scales", 128)):
        assert parity[field] == count
    assert b"lowered connected 64x64: C1 0 codes 0 scales; C2 0 codes 0 scales" in (
        EVIDENCE / "parity/spike.log").read_bytes()
    source_log = (EVIDENCE / "reference/source_baseline/spike.log").read_bytes()
    assert b"fp8 chain test PASSED" in source_log
