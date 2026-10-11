"""Audit the public-object replay of Nicolas's three same-format LUT sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)
from tools.qualify_nicolas_symmetric_lut_public_suite import CASES, SCHEMA


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_symmetric_lut_public_4e30dcf_266c593"
COMPILER = "4e30dcfafa7c81dfaec0c49e2d056d14e037ba7a"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_same_format_lut_sources_have_complete_public_object_replays() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hashes = {entry["name"]: entry["source_sha256"]
                     for entry in inventory["entries"]}
    assert index["schema"] == SCHEMA
    assert index["status"] == "all_three_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == COMPILER
    assert index["rtl_revision"] == RTL_REVISION
    assert index["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert index["mxq_revision"] == MXQ_REVISION
    assert index["selected_cases"] == list(CASES)
    assert [row["case"] for row in index["cases"]] == list(CASES)
    assert index["matched_sources"] == 3
    assert index["total_bf16_outputs_checked"] == 12288

    for row in index["cases"]:
        case = CASES[row["case"]]
        path = ARCHIVE / row["case"]
        receipt = json.loads((path / "receipt.json").read_text())
        dispatch = json.loads((path / "object/compile_manifest.json").read_text())
        obj = json.loads((path / "object/object_manifest.json").read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                               f"{case.profile}.json")
        source = case.source.removesuffix(".c")

        assert row["source_program"] == source
        assert row["source_driver_sha256"] == case.source_sha256 == source_hashes[source]
        assert row["source_header_sha256"] == case.header_sha256
        assert row["receipt_sha256"] == _sha(path / "receipt.json")
        assert row["object_sha256"] == receipt["public_object_sha256"] == _sha(path / "object/mx_issue.o")
        assert row["elf_sha256"] == receipt["elf_sha256"] == _sha(path / "physical/asymmetric_program.elf")
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"] == _sha(path / "physical/spike.log")
        assert row["compared_bf16_outputs"] == receipt["compared_bf16_outputs"] == 4096
        assert row["profile_name"] == receipt["profile_name"] == case.profile
        assert row["profile_sha256"] == receipt["profile_sha256"] == profile_sha256(profile)
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
        assert receipt["mxq_revision"] == MXQ_REVISION
        assert receipt["rtl_revision"] == RTL_REVISION
        assert receipt["activation_projection"] == "lut"
        assert receipt["public_object_dispatch_sha256"] == _sha(path / "object/compile_manifest.json")
        assert receipt["public_object_manifest_sha256"] == _sha(path / "object/object_manifest.json")
        assert receipt["physical_program_sha256"] == _sha(path / "object/physical_program.json")
        assert dispatch["lowering_family"] == "asymmetric_source"
        assert dispatch["object_sha256"] == obj["object_sha256"] == row["object_sha256"]
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert any(slot["name"] == "output_bf16" and slot["role"] == "write"
                   for slot in obj["buffer_abi"])
        assert "0 BF16 mismatches" in (path / "physical/spike.log").read_text()
        for name, digest in receipt["files_sha256"].items():
            assert digest == _sha(path / name)
        for name, digest in receipt["physical_inputs_sha256"].items():
            assert digest == _sha(path / "physical" / name)
        for name, digest in obj["resource_sha256"].items():
            assert digest == _sha(path / "physical" / f"{name}.bin")
        assert "mx_issue(" in (path / "physical/mx_driver.c").read_text()
