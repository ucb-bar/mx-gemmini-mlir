"""Verify archived unqualified accumulator objects and their source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.compile_nicolas_accumulator_readout import CASES_HARDWARE
from tools.qualify_nicolas_plain_matrix_object import CASES, RTL_REVISION


ARCHIVE = (Path(__file__).resolve().parents[1] /
           "docs/evidence/nicolas_accumulator_readout_objects_266c593")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("case", CASES_HARDWARE)
def test_hardware_command_object_is_data_free_and_unqualified(case: str) -> None:
    directory = ARCHIVE / case
    audit = json.loads((directory / "accumulator_command_audit.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    manifest = json.loads((directory / "object/object_manifest.json").read_text())
    assert audit["rtl_revision"] == RTL_REVISION
    assert audit["source_driver_sha256"] == CASES[case].source_sha256
    assert audit["status"] == "accumulator_compute_and_mvout_geometry_matched"
    assert audit["numerical_qualification"] == "unqualified"
    assert audit["compiler_transport"] == "rocket_rocc"
    assert audit["source_hardware_transport"] == "mx_mmio_gateway"
    assert audit["physical_program_sha256"] == _sha(directory / "object/physical_program.json")
    assert audit["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert audit["bound_mlir_sha256"] == _sha(directory / "accumulator_bound.mlir")
    assert audit["object_manifest_sha256"] == _sha(directory / "object/object_manifest.json")
    assert manifest["mode"] == physical["mode"] == "rtl_accumulator"
    assert manifest["execution_scope"] == "hardware_accumulator_commands_only"
    assert manifest["stock_spike_supported"] is False
    assert manifest["hardware_numerical_qualification"] == "unqualified"
    assert manifest["allocated_data_section_bytes"] == 0
    assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
    assert physical["source_golden_preserving"] is False
    assert physical["plan"]["hardware_numerical_qualification"] == "unqualified"
    assert physical["plan"]["readout_source_sha256"] == CASES[case].source_sha256
    assert audit["mvout_commands"] == CASES_HARDWARE[case][1]
