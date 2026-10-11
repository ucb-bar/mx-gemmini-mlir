"""Check the two source-bound I-chunk objects and their Spike evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.chunked_i import bind_i_chunks, selected_i_chunks
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp8_chunked_public_fbe26eb_266c593"
COMPILER = "fbe26ebd55d558460b9b6c08ddc969c4e607fcd3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("chunks", [2, 4])
def test_source_and_compiler_chunked_output_and_commands(chunks: int) -> None:
    case = CASES[f"fp8_128x128x128_chunked{chunks}"]
    directory = EVIDENCE / f"chunk{chunks}"
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "chunk_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    object_manifest = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (directory / "payload_bound.mlir").read_text()

    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["compiler_revision"] == COMPILER
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == case.source_sha256
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert manifest["source_driver_sha256"] == case.source_sha256
    assert len(resources["golden_bf16"]) == 32768
    assert selected_i_chunks(mlir, profile, manifest) == chunks
    assert receipt["i_chunks"] == physical["plan"]["i_chunks"] == chunks
    assert receipt["outputs_checked"] == 16384
    assert receipt["mismatches"] == 0
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert "fp8 WS chunked matmul test PASSED (no mismatches)." in (
        directory / "source_baseline/spike.log").read_text()
    assert f"0 mismatches / 16384 BF16 values" in (
        directory / "run/spike.log").read_text()
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["chunk_equivalence_sha256"] == _sha(
        directory / "chunk_equivalence.json")
    assert audit["source_chunk_count"] == audit["compiler_chunk_count"] == chunks
    assert audit["scale_uploads"] == 2 * chunks
    assert audit["accumulator_bank_toggle_bits"] == [0x100] * chunks
    assert [step["command"].get("funct") for step in physical["steps"]
            if step["phase"] == "compute_chunk"] == [9, 24, 8] * chunks + [None]
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")


def test_chunk_selection_rejects_changed_source_or_unbound_attribute() -> None:
    directory = EVIDENCE / "chunk2"
    manifest, _ = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (directory / "payload_bound.mlir").read_text()
    with pytest.raises(ValueError, match="not bound"):
        selected_i_chunks(mlir, profile,
                          {**manifest, "source_driver_sha256": "0" * 64})
    with pytest.raises(ValueError, match="pinned source"):
        selected_i_chunks(mlir.replace('mx.i_chunks = "2"',
                                       'mx.i_chunks = "4"'), profile, manifest)
    with pytest.raises(ValueError, match="one source-bound"):
        bind_i_chunks(mlir, profile, manifest, 2)


def test_direct_2d_scale_source_uses_the_same_two_chunk_object() -> None:
    directory = ROOT / "docs/evidence/nicolas_fp8_chunked_2d_public_287593c_266c593"
    case = CASES["fp8_128x128x128_chunked_2d"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "chunk_equivalence.json").read_text())
    manifest, _ = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    assert receipt["compiler_revision"] == "287593c43879533e03f7dbcd6f82b6f553e74804"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 16384
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "fp8 WS chunked matmul test PASSED (no mismatches)." in (
        directory / "source_baseline/spike.log").read_text()
    assert selected_i_chunks((directory / "payload_bound.mlir").read_text(),
                             profile, manifest) == 2
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["source_chunk_count"] == audit["compiler_chunk_count"] == 2
    assert audit["scale_uploads"] == 4
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o") == _sha(
        EVIDENCE / "chunk2/object/mx_issue.o")
