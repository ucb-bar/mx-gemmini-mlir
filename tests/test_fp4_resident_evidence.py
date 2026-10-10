"""Verify archived model2MLIR, RoCC object, and Nicolas Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_fp4_connected_resident_266c593"
ARCHIVE_128 = ROOT / "docs/evidence/nicolas_fp4_connected_resident_128_266c593"


def _read(path: str) -> bytes:
    raw = (ARCHIVE / path).read_bytes()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def _read_128(path: str) -> bytes:
    raw = (ARCHIVE_128 / path).read_bytes()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def test_connected_fp4_archive_integrity_and_spike_results():
    index = json.loads(_read("index.json"))
    assert index["schema"] == "mx_gemmini.nicolas_fp4_connected_resident_archive.v1"
    assert index["status"] == "source_and_compiler_matched_on_pinned_spike"
    assert index["compared"] == {
        "c1_fp4_codes": 4096, "c2_fp4_codes": 4096,
        "c1_e8m0_scales": 128, "c2_e8m0_scales": 128}
    for path, record in index["files"].items():
        data = _read(path)
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert record["compressed"] == path.endswith(".gz")
    capture = json.loads(_read("capture/receipt.json"))
    chain = json.loads(_read("chain/receipt.json"))
    obj = json.loads(_read("object/object_manifest.json"))
    dispatch = json.loads(_read("object/compile_manifest.json"))
    fresh = json.loads(_read("fresh_checkout.json"))
    assert capture["status"] == "two_site_frontend_handoff_only"
    assert [site["format"] for site in capture["sites"]] == ["mxfp4", "mxfp4"]
    assert capture["opaque_calls"] == []
    assert chain["status"] == index["status"]
    assert chain["source_spike"]["matched"]
    assert chain["compiler_spike"]["matched"]
    assert chain["allocated_data_section_bytes"] == 0
    assert chain["bound_mlir_sha256"] == index["files"]["chain/connected.mlir"]["sha256"]
    assert obj["precision"] == "fp4_e2m1"
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["bound_mlir_sha256"] == chain["bound_mlir_sha256"]
    assert dispatch["lowering_family"] == "resident_pair"
    assert dispatch["compiler_revision"] == index["dispatcher_revision"]
    assert dispatch["object_sha256"] == obj["object_sha256"]
    assert {entry["slot"] for entry in obj["buffer_abi"]} == {
        "a1_activation", "a1_scales", "b1_weight", "b1_scales",
        "b2_weight", "b2_scales", "c1_scales", "c1_tiled_observed",
        "c2_scales", "c2_tiled"}
    assert b"fp4 chain test PASSED" in _read("chain/source_spike.log")
    assert b"fp4 chain test PASSED" in _read("chain/compiled_spike.log")
    assert fresh["schema"] == "mx_gemmini.nicolas_fp4_connected_resident_fresh_checkout.v1"
    assert fresh["status"] == "fresh_checkout_reproduced_capture_commands_object_elf_and_spike"
    assert fresh["source_spike_exit_code"] == fresh["compiler_spike_exit_code"] == 0
    for archive_path, fresh_path in {
        "capture/source.mlir.gz": "capture/nicolas_fp4_chain.model2mlir.mlir",
        "capture/handoff.mlir.gz": "capture/nicolas_fp4_chain.handoff.mlir",
        "capture/profile_bound.mlir.gz": "capture/nicolas_fp4_chain.profile_bound.mlir",
        "capture/quantization_manifest.json": "capture/quantization_manifest.json",
        "chain/connected.mlir": "chain/connected.mlir",
        "chain/physical_program.json.gz": "chain/physical_program.json",
        "chain/mx_issue.c.gz": "chain/mx_issue.c",
        "chain/mx_issue.o.gz": "chain/mx_issue.o",
        "chain/source_spike.log": "chain/source/spike.log",
        "chain/compiled_spike.log": "chain/compiled/spike.log",
        "chain/source.elf.gz": "chain/source/program.elf",
        "chain/compiled.elf.gz": "chain/compiled/program.elf",
        "object/mx_issue.c.gz": "object/mx_issue.c",
        "object/mx_issue.o.gz": "object/mx_issue.o",
        "object/physical_program.json.gz": "object/physical_program.json",
    }.items():
        assert fresh["checks_sha256"][fresh_path] == index["files"][archive_path]["sha256"]


def test_connected_fp4_128_archive_and_full_output_spike():
    index = json.loads(_read_128("index.json"))
    assert index["schema"] == "mx_gemmini.nicolas_fp4_connected_resident_128_archive.v1"
    assert index["status"] == "source_and_compiler_matched_on_pinned_spike"
    assert index["compared"] == {
        "c1_fp4_codes": 16384, "c2_fp4_codes": 16384,
        "c1_e8m0_scales": 512, "c2_e8m0_scales": 512}
    for path, record in index["files"].items():
        data = _read_128(path)
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    capture = json.loads(_read_128("capture/receipt.json"))
    chain = json.loads(_read_128("chain/receipt.json"))
    obj = json.loads(_read_128("object/object_manifest.json"))
    dispatch = json.loads(_read_128("object/compile_manifest.json"))
    assert capture["matrix_dim"] == chain["matrix_dim"] == 128
    assert [site["format"] for site in capture["sites"]] == ["mxfp4", "mxfp4"]
    assert capture["opaque_calls"] == []
    assert chain["status"] == index["status"]
    assert chain["source_spike"]["matched"]
    assert chain["compiler_spike"]["matched"]
    assert chain["compared_c1_fp4_codes"] == chain["compared_c2_fp4_codes"] == 16384
    assert chain["compared_c1_e8m0_scales"] == chain["compared_c2_e8m0_scales"] == 512
    assert chain["allocated_data_section_bytes"] == 0
    assert obj["precision"] == "fp4_e2m1"
    assert obj["shape_mnk"] == [128, 128, 128]
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["allocated_data_section_bytes"] == 0
    assert dispatch["lowering_family"] == "resident_pair"
    assert dispatch["compiler_revision"] == index["compiler_revision"]
    assert dispatch["object_sha256"] == obj["object_sha256"]
    assert obj["bound_mlir_sha256"] == chain["bound_mlir_sha256"]
    assert b"fp4 chain test PASSED" in _read_128("chain/source_spike.log")
    assert b"fp4 chain test PASSED" in _read_128("chain/compiled_spike.log")
