"""Verify archived source and generated-object Spike fallback qualifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("key,count", (
    ("fp8_64x64x64_dram_mvout_spike", 4096),
    ("fp4_64x64x64_dram_mvout_spike", 4096),
    ("fp8_128x128x256_dram_mvout_spike", 16384),
))
def test_compiled_and_source_spike_fallback_match(key: str, count: int) -> None:
    case = CASES[key]
    directory = EVIDENCE / key
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "smem_readout_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    obj = json.loads((directory / "object/object_manifest.json").read_text())

    assert receipt["compiler_revision"] == "6ed3fcba1ac4c29d66575799d7aac0088e36e4d4"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["source_driver_sha256"] == case.source_sha256
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["qualification_scope"] == "SPIKE_SIM_scratchpad_fallback_only"
    assert receipt["hardware_accumulator_mvout_qualified"] is False
    assert receipt["outputs_checked"] == count and receipt["mismatches"] == 0
    assert receipt["source_baseline"]["source_execution_path"] == (
        "SPIKE_SIM_scratchpad_fallback")
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == count
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert "fp8 WS matmul test PASSED (no mismatches)." in (
        directory / "source_baseline/spike.log").read_text()
    assert f"0 mismatches / {count} BF16 values" in (
        directory / "run/spike.log").read_text()
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert obj["allocated_data_section_bytes"] == 0
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert audit["readout_count"] == count * 2 // (16 * 16)
    assert physical["plan"]["c_spad_dest"] == 0
    assert physical["plan"]["readout_source_sha256"] == case.source_sha256
    assert len([step for step in physical["steps"] if
                step["phase"] == "compute" and step["command"].get("funct") == 8]) == (
                    case.shape[2] // case.tile[2])
