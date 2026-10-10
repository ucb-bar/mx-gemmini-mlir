"""Check archived batched decode source, compiler, and Spike identities."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from tools.materialize_radiance_batched_gemv_data import CASES
from tools.materialize_radiance_ws_data import GENERATOR_SHA256, SOURCE_REVISION
from tools.qualify_radiance_ws_roster import (
    _prove_weight_read_once, _semantic_manifest_sha,
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION)


EVIDENCE = (Path(__file__).resolve().parents[1] /
            "docs/evidence/radiance_batched_gemv_552a16b")
COMPILER_REVISION = "552a16bd09efa6c4eb3a794274b12d39c3d0e790"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_batched_gemv_source_and_spike_reproduction() -> None:
    index_bytes = (EVIDENCE / "index.json").read_bytes()
    assert index_bytes == (EVIDENCE / "index_repro.json").read_bytes()
    index = json.loads(index_bytes)
    assert index["schema"] == "mx_gemmini.radiance_batched_gemv_roster.v1"
    assert index["status"] == "four_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == COMPILER_REVISION
    assert (index["source_revision"], index["model2mlir_revision"],
            index["mxq_revision"], index["rtl_revision"]) == (
        SOURCE_REVISION, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION)
    fresh = _read(EVIDENCE / "fresh_data_materialization.json")
    verified = _read(EVIDENCE / "data_materialization.json")
    assert fresh["source_revision"] == verified["source_revision"] == SOURCE_REVISION
    assert fresh["generator_sha256"] == verified["generator_sha256"] == GENERATOR_SHA256
    assert len(fresh["generated"]) == 8
    assert verified["generated"] == []
    assert index["data_materialization_sha256"] == _sha(
        (EVIDENCE / "data_materialization.json").read_bytes())
    assert len(index["rows"]) == 4

    for row, (directory, precision, m, driver_sha, data_sha) in zip(
            index["rows"], CASES):
        root = EVIDENCE / directory
        assert (row["directory"], row["precision"], row["shape_mnk"],
                row["tile_mnk"]) == (directory, precision, [m, 128, 2048],
                                      [m, 128, 128])
        assert (row["driver_sha256"], row["data_sha256"]) == (driver_sha, data_sha)
        assert (row["compared_bf16_outputs"], row["k_waves"],
                row["weight_tile_reads"]) == (m * 128, 16, 16)
        assert row["weight_bytes_covered_once"] == (
            2048 * (128 if precision == "fp8" else 64))
        assert row["frontend_mlir_sha256"] == _sha((root / "model2mlir.mlir").read_bytes())
        assert row["bound_mlir_sha256"] == _sha((root / "profile_bound.mlir").read_bytes())
        assert row["payload_bound_mlir_sha256"] == _sha(
            (root / "payload_bound.mlir").read_bytes())
        front = [_read(root / f"frontend_{label}.json") for label in ("first", "repro")]
        assert front[0] == front[1]
        assert row["frontend_receipt_sha256"] == _sha(
            (root / "frontend_first.json").read_bytes())
        assert front[0]["source_driver_sha256"] == driver_sha
        assert front[0]["source_data_header_sha256"] == data_sha
        assert not front[0]["opaque_calls"]
        assert front[0]["target_binding"]["profile_sha256"] == index["profile_sha256"]
        program_bytes = gzip.decompress((root / "physical_program.json.gz").read_bytes())
        assert row["physical_program_sha256"] == _sha(program_bytes)
        assert _prove_weight_read_once(
            json.loads(program_bytes), m=m, n=128, k=2048,
            tile_k=128, precision=precision)["weight_dma_commands"] == (
                row["weight_dma_commands"])
        issuer = gzip.decompress((root / "mx_issue.c.gz").read_bytes())
        assert row["files_sha256"]["mx_issue.c"] == _sha(issuer)
        for name in ("mx_driver.c", "mx_data.S"):
            assert row["files_sha256"][name] == _sha((root / name).read_bytes())
        receipts = [_read(root / f"spike_{label}.json") for label in ("first", "repro")]
        assert _semantic_manifest_sha(receipts[0]) == _semantic_manifest_sha(receipts[1])
        assert _semantic_manifest_sha(receipts[0]) == row["spike_manifest_semantic_sha256"]
        for receipt in receipts:
            assert receipt["status"] == "source_golden_matched_on_pinned_spike"
            assert receipt["spike_exit_code"] == 0
            assert receipt["compared_bf16_outputs"] == m * 128
            assert receipt["source_driver_sha256"] == driver_sha
            assert receipt["source_header_sha256"] == data_sha
            assert receipt["files_sha256"] == row["files_sha256"]
            assert receipt["object_sha256"] == row["object_sha256"]
            assert receipt["elf_sha256"] == row["elf_sha256"]
            assert receipt["extension_sha256"] == row["extension_sha256"]
            assert receipt["spike_log_sha256"] == row["spike_log_sha256"]
        assert row["spike_log_sha256"] == _sha((root / "spike.log").read_bytes())
