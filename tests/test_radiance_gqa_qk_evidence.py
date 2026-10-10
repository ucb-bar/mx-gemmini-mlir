"""Keep the source overflow and candidate Spike result separately auditable."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_attention_qk import (
    DRIVER_SHA256, HEADER_SHA256, SOURCE_REVISION, SUBMODULE_REVISION)
from mx_gemmini_support.source_payload import (
    ATTENTION_QK_CANDIDATE_ORIGIN, validate_attention_qk_candidate)
from tools.qualify_radiance_ws_roster import (_semantic_manifest_sha,
                                              MODEL2MLIR_REVISION, MXQ_REVISION,
                                              RTL_REVISION)


EVIDENCE = (Path(__file__).resolve().parents[1] /
            "docs/evidence/radiance_gqa_qk_candidate_5baebbe")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_original_qk_source_overflows_and_candidate_reproduces():
    index_bytes = (EVIDENCE / "index.json").read_bytes()
    assert index_bytes == (EVIDENCE / "index_repro.json").read_bytes()
    index = json.loads(index_bytes)
    assert index["schema"] == "mx_gemmini.radiance_gqa_qk_candidate.v1"
    assert index["status"] == "shifted_source_qk_candidate_matched_on_pinned_spike"
    assert "no full attention or original source golden parity" in index["scope"]
    assert index["compiler_revision"] == "5baebbef724c42c8ef47bebe20a1656e859a5806"
    assert (index["source_revision"], index["submodule_revision"],
            index["model2mlir_revision"], index["mxq_revision"],
            index["rtl_revision"]) == (
        SOURCE_REVISION, SUBMODULE_REVISION, MODEL2MLIR_REVISION,
        MXQ_REVISION, RTL_REVISION)
    assert index["source_driver_sha256"] == DRIVER_SHA256
    assert index["source_header_sha256"] == HEADER_SHA256
    assert index["original_source_products_over_448"] == 244154
    assert index["compared_bf16_outputs"] == 4096
    assert index["frontend_mlir_sha256"] == _sha((EVIDENCE / "model2mlir.mlir").read_bytes())
    assert index["bound_mlir_sha256"] == _sha((EVIDENCE / "payload_bound.mlir").read_bytes())

    manifest_bytes = (EVIDENCE / "bundle_manifest.json").read_bytes()
    assert index["bundle_manifest_sha256"] == _sha(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    assert manifest["origin"] == ATTENTION_QK_CANDIDATE_ORIGIN
    validate_attention_qk_candidate(manifest)
    assert manifest["source_derivation"] == index["policy"]
    assert manifest["source_header_sha256"] == HEADER_SHA256
    for name, descriptor in manifest["resources"].items():
        data = (EVIDENCE / f"{name}.bin").read_bytes()
        assert _sha(data) == descriptor["sha256"]
        assert len(data) == descriptor["bytes"]
    assert (EVIDENCE / "output_scales.bin").read_bytes() == bytes([127] * 128)

    program = gzip.decompress((EVIDENCE / "physical_program.json.gz").read_bytes())
    assert index["physical_program_sha256"] == _sha(program)
    assert json.loads(program)["shape_mnk"] == [64, 64, 64]
    assert index["files_sha256"]["mx_issue.c"] == _sha(
        gzip.decompress((EVIDENCE / "mx_issue.c.gz").read_bytes()))
    for name in ("mx_driver.c", "mx_data.S"):
        assert index["files_sha256"][name] == _sha((EVIDENCE / name).read_bytes())
    assert index["spike_log_sha256"] == _sha((EVIDENCE / "spike.log").read_bytes())
    receipts = [_read(EVIDENCE / f"spike_{label}.json") for label in ("first", "repro")]
    assert _semantic_manifest_sha(receipts[0]) == _semantic_manifest_sha(receipts[1])
    assert _semantic_manifest_sha(receipts[0]) == index["spike_manifest_semantic_sha256"]
    for receipt in receipts:
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compared_bf16_outputs"] == 4096
        assert receipt["files_sha256"] == index["files_sha256"]
        assert receipt["object_sha256"] == index["object_sha256"]
        assert receipt["elf_sha256"] == index["elf_sha256"]
        assert receipt["spike_log_sha256"] == index["spike_log_sha256"]

    original = EVIDENCE / "original_source_probe"
    failure = _read(original / "spike_failure.json")
    assert failure["status"] == "source_golden_failed_on_pinned_spike"
    assert failure["spike_exit_code"] == 1
    assert failure["compared_bf16_outputs"] == 4096
    assert failure["source_header_sha256"] == HEADER_SHA256
    assert failure["spike_log_sha256"] == _sha((original / "spike_failure.log").read_bytes())
    assert "4096 BF16 mismatches" in (original / "spike_failure.log").read_text()
    assert "got=0x7fc0" in (original / "spike_failure.log").read_text()
    raw = _read(original / "bundle_manifest.json")
    assert raw["origin"] == "radiance_source_header_specialization"
    for name, digest in index["source_arrays_sha256"].items():
        assert digest == _sha((original / f"{name}.bin").read_bytes())
    assert failure["files_sha256"]["physical_program.json"] == _sha(
        gzip.decompress((original / "physical_program.json.gz").read_bytes()))
