"""Verify archived model2MLIR, RoCC object, and Nicolas Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_fp4_connected_resident_266c593"


def _read(path: str) -> bytes:
    raw = (ARCHIVE / path).read_bytes()
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
