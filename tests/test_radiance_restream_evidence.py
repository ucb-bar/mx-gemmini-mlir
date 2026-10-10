"""Check the pinned four-pass re-stream source and Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from tools.qualify_radiance_restream import (DATA_SHA256, DRIVER_SHA256,
                                             prove_restream_weight_reads)
from tools.qualify_radiance_ws_roster import (_semantic_manifest_sha,
                                              MODEL2MLIR_REVISION, MXQ_REVISION,
                                              RTL_REVISION)
from tools.materialize_radiance_ws_data import SOURCE_REVISION


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_restream_1df2c5f"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_restream_compiler_program_and_spike_reproduction() -> None:
    assert (EVIDENCE / "index.json").read_bytes() == (
        EVIDENCE / "index_repro.json").read_bytes()
    index = _read(EVIDENCE / "index.json")
    assert index["schema"] == "mx_gemmini.radiance_restream_source_parity.v1"
    assert index["status"] == "four_m_tile_restream_source_golden_matched_on_pinned_spike"
    assert index["compiler_revision"] == "1df2c5fe63cd39528dcc975f50f9e7bf47834525"
    assert (index["source_revision"], index["model2mlir_revision"],
            index["mxq_revision"], index["rtl_revision"]) == (
        SOURCE_REVISION, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION)
    assert (index["driver_sha256"], index["data_sha256"]) == (
        DRIVER_SHA256, DATA_SHA256)
    assert (index["shape_mnk"], index["tile_mnk"],
            index["compared_bf16_outputs"]) == ([256, 64, 2048], [64, 64, 64], 16384)
    assert (index["m_tiles"], index["k_waves_per_m_tile"],
            index["weight_tile_reads"], index["weight_dma_commands"],
            index["weight_bytes_per_pass"], index["weight_bytes_transferred"]) == (
        4, 32, 128, 2048, 131072, 524288)
    read_once = _read(ROOT / "docs/evidence/radiance_ws_read_once_1a7bc9c/index.json")
    assert index["profile_sha256"] == read_once["profile_sha256"]
    assert index["data_sha256"] != read_once["rows"][0]["data_sha256"]
    assert index["weight_tile_reads"] == 4 * read_once["rows"][0]["weight_tile_reads"]
    assert index["frontend_mlir_sha256"] == _sha((EVIDENCE / "model2mlir.mlir").read_bytes())
    assert index["bound_mlir_sha256"] == _sha((EVIDENCE / "profile_bound.mlir").read_bytes())
    assert index["payload_bound_mlir_sha256"] == _sha(
        (EVIDENCE / "payload_bound.mlir").read_bytes())
    frontend = [_read(EVIDENCE / f"frontend_{label}.json")
                for label in ("first", "repro")]
    assert frontend[0] == frontend[1]
    assert index["frontend_receipt_sha256"] == _sha(
        (EVIDENCE / "frontend_first.json").read_bytes())
    assert frontend[0]["source_driver_sha256"] == DRIVER_SHA256
    assert frontend[0]["source_data_header_sha256"] == DATA_SHA256
    assert not frontend[0]["opaque_calls"]
    program = gzip.decompress((EVIDENCE / "physical_program.json.gz").read_bytes())
    assert index["physical_program_sha256"] == _sha(program)
    assert prove_restream_weight_reads(json.loads(program)) == {
        key: index[key] for key in ("m_tiles", "k_waves_per_m_tile",
                                   "weight_tile_reads", "weight_dma_commands",
                                   "weight_bytes_per_pass", "weight_bytes_transferred")}
    assert index["files_sha256"]["mx_issue.c"] == _sha(
        gzip.decompress((EVIDENCE / "mx_issue.c.gz").read_bytes()))
    for name in ("mx_driver.c", "mx_data.S"):
        assert index["files_sha256"][name] == _sha((EVIDENCE / name).read_bytes())
    receipts = [_read(EVIDENCE / f"spike_{label}.json")
                for label in ("first", "repro")]
    assert _semantic_manifest_sha(receipts[0]) == _semantic_manifest_sha(receipts[1])
    assert index["spike_manifest_semantic_sha256"] == _semantic_manifest_sha(receipts[0])
    for receipt in receipts:
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compared_bf16_outputs"] == 16384
        assert receipt["source_driver_sha256"] == DRIVER_SHA256
        assert receipt["source_header_sha256"] == DATA_SHA256
        for key in ("files_sha256", "object_sha256", "elf_sha256",
                    "extension_sha256", "spike_log_sha256"):
            assert receipt[key] == index[key]
    assert index["spike_log_sha256"] == _sha((EVIDENCE / "spike.log").read_bytes())
