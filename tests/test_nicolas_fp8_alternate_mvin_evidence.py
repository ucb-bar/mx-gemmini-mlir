"""Audit the pinned source and public-object alternate FP8 transfer replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp8_alternate_mvin_compiled_5a2dd35_266c593"
BASELINE = ROOT / "docs/evidence/nicolas_direct_matrix_suite_9df3384_266c593/fp8_32x32x32/object"
COMPILER = "5a2dd3520a7098a8ebf147a107b6e79ee8747e44"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_compiler_transfer_stream_and_output_match() -> None:
    case = CASES["fp8_32x32x32_alternate_mvin"]
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    transfer = json.loads((EVIDENCE / "transfer_equivalence.json").read_text())
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    listed = next(entry for entry in inventory["entries"] if entry["name"] == "matmul_tiled_fp8")
    assert listed["source_sha256"] == case.source_sha256
    assert any("nicolas_fp8_alternate_mvin_compiled_5a2dd35" in ref["path"]
               for ref in listed["evidence_references"])
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["compiler_revision"] == COMPILER
    assert receipt["source_driver_sha256"] == case.source_sha256
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["outputs_checked"] == 1024
    assert receipt["mismatches"] == 0
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 1024
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        EVIDENCE / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        EVIDENCE / "source_baseline/spike.log")
    assert "fp8 WS matmul test PASSED (no mismatches)" in (
        EVIDENCE / "source_baseline/spike.log").read_text()
    assert "0 mismatches / 1024 BF16 values" in (EVIDENCE / "run/spike.log").read_text()
    assert receipt["spike_log_sha256"] == _sha(EVIDENCE / "run/spike.log")
    assert receipt["elf_sha256"] == _sha(EVIDENCE / "run/mx_program.elf")
    assert receipt["transfer_equivalence_sha256"] == _sha(
        EVIDENCE / "transfer_equivalence.json")
    assert transfer["source_driver_sha256"] == case.source_sha256
    assert transfer["source_transfer_pairs"] == transfer["compiler_transfer_pairs"]
    assert transfer["compiler_transfer_pairs"]["move_weight"] == [
        [0, 16320], [16, 16336], [512, 16352], [528, 16368]]
    assert transfer["physical_program_sha256"] == _sha(
        EVIDENCE / "object/physical_program.json")
    assert receipt["object_sha256"] == transfer["object_sha256"] == _sha(
        EVIDENCE / "object/mx_issue.o") == _sha(BASELINE / "mx_issue.o")
    assert receipt["source_mlir_sha256"] == _sha(EVIDENCE / "model2mlir.mlir")
    assert receipt["bound_mlir_sha256"] == _sha(EVIDENCE / "payload_bound.mlir")
    manifest = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    assert manifest["allocated_data_section_bytes"] == 0
