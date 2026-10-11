"""Verify the archived source-bound public VPU sequence artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_vpu_chain_public_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nicolas_vpu_chain_archive() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_vpu_chain_public_archive.v1"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    source_receipt = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593/receipt.json"
    assert _sha(source_receipt) == index["source_receipt_sha256"]
    source = json.loads(source_receipt.read_text())
    assert source["status"] == "all_29_source_vpu_checks_matched_on_pinned_spike"
    assert "chain" in source["checks"]
    assert [row["profile_name"] for row in index["rows"]] == [
        "e4m3_vpu", "e4m3_fp4_vpu"]
    assert (ARCHIVE / "e4m3_vpu/receipt.json").read_bytes() == (
        ARCHIVE / "reproduction_receipt.json").read_bytes()
    assert _sha(ARCHIVE / "reproduction_receipt.json") == (
        index["reproduction_receipt_sha256"])
    receipts = []
    for row in index["rows"]:
        directory = ARCHIVE / row["profile_name"]
        receipt_path = ARCHIVE / row["receipt"]
        assert _sha(receipt_path) == row["receipt_sha256"]
        receipt = json.loads(receipt_path.read_text())
        receipts.append(receipt)
        assert receipt["schema"] == "mx_gemmini.nicolas_vpu_chain_public_spike.v1"
        assert receipt["status"] == "source_chain_matched_by_public_object_on_pinned_spike"
        assert receipt["source_sha256"] == index["source_sha256"]
        assert receipt["reference_sha256"] == index["reference_sha256"]
        assert receipt["source_receipt_sha256"] == index["source_receipt_sha256"]
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["model2mlir_revision"] == index["model2mlir_revision"]
        assert receipt["model2mlir_source_closure_sha256"] == (
            index["model2mlir_source_closure_sha256"])
        assert receipt["profile_sha256"] == row["profile_sha256"]
        assert receipt["spike_exit_code"] == receipt["mismatches"] == 0
        assert (receipt["compared_middle_bf16"],
                receipt["compared_result_bf16"]) == (512, 128)
        for field, path in (
                ("frontend_mlir_sha256", "frontend.mlir"),
                ("bound_mlir_sha256", "bound.mlir"),
                ("abi_sha256", "abi.json"),
                ("driver_sha256", "mx_driver.c"),
                ("object_sha256", "object/mx_issue.o"),
                ("object_manifest_sha256", "object/object_manifest.json"),
                ("physical_program_sha256", "object/physical_program.json"),
                ("elf_sha256", "vpu_chain.elf"),
                ("spike_log_sha256", "spike.log")):
            assert _sha(directory / path) == receipt[field], path
        assert "0 middle mismatches, 0 result mismatches" in (
            directory / "spike.log").read_text()
        physical = json.loads((directory / "object/physical_program.json").read_text())
        assert [op["kind"] for op in physical["operations"]] == [
            "add", "muls", "rmax"]
        assert [op["funct"] for op in physical["commands"]].count(33) == 3
        manifest = json.loads((directory / "object/object_manifest.json").read_text())
        assert manifest["schema"] == "mx_gemmini.vpu_sequence_linkable_object.v1"
        assert manifest["object_sha256"] == receipt["object_sha256"]
        assert manifest["allocated_data_section_bytes"] == 0
        assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
        assert manifest["operation_count"] == 3
        frontend = (directory / "frontend.mlir").read_text()
        assert all(marker in frontend for marker in (
            'prov.aten = "aten.add.Tensor"',
            'prov.aten = "aten.mul.Tensor"',
            'prov.aten = "aten.amax.default"'))
    first, second = receipts
    assert first["profile_sha256"] != second["profile_sha256"]
    assert first["bound_mlir_sha256"] != second["bound_mlir_sha256"]
    for field in ("frontend_mlir_sha256", "abi_sha256", "driver_sha256",
                  "object_sha256", "elf_sha256", "spike_log_sha256"):
        assert first[field] == second[field]
