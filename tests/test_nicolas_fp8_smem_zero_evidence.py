"""Audit Nicolas's zero-base scratchpad FP8 source and compiled object."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.smem_readback import selected_smem_zero_readout
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp8_smem_zero_public_64ecacd_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_compiled_zero_base_smem_match() -> None:
    case = CASES["fp8_64x64x64_smem_mvout"]
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    audit = json.loads((EVIDENCE / "smem_readout_equivalence.json").read_text())
    physical = json.loads((EVIDENCE / "object/physical_program.json").read_text())
    obj = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (EVIDENCE / "payload_bound.mlir").read_text()

    assert receipt["compiler_revision"] == "64ecacd1fc95a4cfacd894e355ce2e7567997344"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["outputs_checked"] == 4096 and receipt["mismatches"] == 0
    assert len(resources["golden_bf16"]) == 8192
    assert selected_smem_zero_readout(mlir, profile, manifest)
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["compute_c_scratchpad_row"] == 0
    assert audit["readout_transport"] == "spad_mvout_zero_base"
    assert audit["readout_rows"] == list(range(0, 512, 16))
    assert audit["readout_count"] == 32
    assert audit["physical_program_sha256"] == _sha(
        EVIDENCE / "object/physical_program.json")
    assert receipt["smem_readout_equivalence_sha256"] == _sha(
        EVIDENCE / "smem_readout_equivalence.json")
    assert physical["plan"]["c_spad_dest"] == 0
    assert physical["plan"]["readout_transport"] == "spad_mvout_zero_base"
    assert obj["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(EVIDENCE / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(EVIDENCE / "run/spike.log")
    assert "0 mismatches / 4096 BF16 values" in (
        EVIDENCE / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 4096
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        EVIDENCE / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        EVIDENCE / "source_baseline/spike.log")
    assert "fp8 WS matmul test PASSED (no mismatches)." in (
        EVIDENCE / "source_baseline/spike.log").read_text()

    with pytest.raises(ValueError, match="not bound"):
        selected_smem_zero_readout(
            mlir, profile, {**manifest, "source_driver_sha256": "0" * 64})
    with pytest.raises(ValueError, match="not bound"):
        selected_smem_zero_readout(mlir.replace(
            'mx.smem_zero_readout = "source64"',
            'mx.smem_zero_readout = "other"'), profile, manifest)
