"""Check the archived complete VPU source replay, not only its summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_vpu_ops_program import BASELINE


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_ops_program_public_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_profile_full_vpu_program_archive() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    source = json.loads(BASELINE.read_text())
    assert index["schema"] == "mx_gemmini.nicolas_vpu_ops_public_program_archive.v1"
    assert index["status"] == "two_profile_full_vpu_program_source_reference_matched"
    assert index["source_sha256"] == source["source_sha256"]
    assert (index["operation_count"], index["snapshot_count"],
            index["source_check_count"], index["bf16_values_checked_per_profile"]) == (
                30, 30, 29, 13056)
    assert index["same_profile_reproduction_exact"] is True
    assert _sha(EVIDENCE / "reproduction_receipt.json") == index[
        "same_profile_second_receipt_sha256"]
    assert json.loads((EVIDENCE / "reproduction_receipt.json").read_text()) == json.loads(
        (EVIDENCE / "e4m3_vpu/receipt.json").read_text())
    assert len(index["rows"]) == 2
    receipts = []
    for row in index["rows"]:
        directory = EVIDENCE / row["name"]
        assert all(_sha(directory / name) == digest
                   for name, digest in row["files_sha256"].items())
        receipt = json.loads((directory / "receipt.json").read_text())
        manifest = json.loads((directory / "object/object_manifest.json").read_text())
        physical = json.loads((directory / "object/physical_program.json").read_text())
        assert receipt["source_sha256"] == source["source_sha256"]
        assert receipt["reference_sha256"] == source["reference_sha256"]
        assert receipt["profile_sha256"] == row["profile_sha256"]
        assert receipt["status"] == "compiler_vpu_program_matched_source_reference_on_pinned_spike"
        assert (receipt["operation_count"], receipt["snapshot_count"],
                receipt["source_check_count"], receipt["bf16_values_checked"],
                receipt["mismatches"]) == (30, 30, 29, 13056, 0)
        assert manifest["buffer_map_schema"] == "mx_gemmini.vpu_sequence_buffer_map.v2"
        assert manifest["allocated_data_section_bytes"] == 0
        assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
        assert sum(command.get("funct") == 33 for command in physical["commands"]) == 30
        assert sum(command.get("kind") == "fence" for command in physical["commands"]) == 26
        lines = (directory / "spike.log").read_text().splitlines()
        assert sum(line.endswith(": 0 mismatches") for line in lines) == 30
        assert "compiled vpu_ops: 0 mismatches across 30 output snapshots" in lines
        receipts.append(receipt)
    assert index["cross_profile_object_elf_log_exact"] is True
    assert receipts[0]["profile_sha256"] != receipts[1]["profile_sha256"]
    assert all(receipts[0][field] == receipts[1][field]
               for field in ("object_sha256", "elf_sha256", "spike_log_sha256"))
