"""Check the archived current-frontend FP6 debug-data qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_ws_generic_portable_a042643_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_generic_fp6_archive_rechecks_all_output_classes() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    binding = json.loads((EVIDENCE / "frontend_binding.json").read_text())
    object_manifest = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_ws_generic_portable_archive.v1"
    assert index["source_sha256"] == receipt["source_driver_sha256"]
    assert index["source_header_sha256"] == receipt["source_header_sha256"]
    assert index["source_transport"] == "fixed_mmio"
    assert index["compiled_transport"] == "rocket_rocc"
    assert index["source_mmio_issue_qualification"] == "not_tested"
    assert (receipt["bf16_values_checked"], receipt["packed_bytes_checked"],
            receipt["e8m0_scales_checked"], receipt["mismatches"]) == (
                16384, 8192, 512, 0)
    assert all(_sha(EVIDENCE / name) == digest
               for name, digest in index["files_sha256"].items())
    assert set(index["files_sha256"]) >= {
        "model2mlir.mlir", "profile_bound.mlir", "payload_bound.mlir",
        "bundle/manifest.json", "bundle/activation_lut.bin", "bundle/weight_lut.bin",
        "bundle/output_lut.bin", "object/mx_issue.o",
        "object/physical_program.json", "run/mx_program.elf", "run/spike.log",
        "receipt.json"}
    assert receipt == json.loads((EVIDENCE / "reproduction_receipt.json").read_text())
    assert index["same_profile_reproduction_exact"] is True
    assert _sha(EVIDENCE / "reproduction_receipt.json") == index[
        "reproduction_receipt_sha256"]
    assert binding["source_mlir_sha256"] == receipt["frontend_mlir_sha256"]
    assert binding["profile_sha256"] == receipt["profile_sha256"]
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert object_manifest["embedded_golden_bytes"] == 0
    assert _sha(EVIDENCE / "object/mx_issue.o") == receipt["object_sha256"]
    assert _sha(EVIDENCE / "run/mx_program.elf") == receipt["elf_sha256"]
    assert _sha(EVIDENCE / "run/spike.log") == receipt["spike_log_sha256"]
    assert "0 bf16 / 16384, 0 packed / 8192, 0 scale / 512 mismatches" in (
        EVIDENCE / "run/spike.log").read_text()
