"""Audit every pinned Nicolas asymmetric source and public-object Spike result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_asym_public_suite import GROUPS, SCHEMA
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_asym_public_suite_e6923e8_266c593"
COMPILER = "e6923e8a774748f55ea3d065f6d3832d6618022d"
REPRESENTATIVES = {
    ("dim8_64x64", "fp4_fp6"), ("dim16_64x64", "e4m3_fp4"),
    ("dim32_64x64", "fp4_fp6"), ("dim16_16x32", "e4m3s_fp4"),
    ("dim16_128x128", "e4m3s_fp4"),
    ("dim32_128x128x256", "fp4_fp6"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_74_asymmetric_sources_have_public_objects_and_full_spike_logs() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    baseline_path = ARCHIVE / "source_inventory_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    inventory = {entry["name"]: entry["source_sha256"]
                 for entry in baseline["entries"]
                 if entry["family"] == "asymmetric_matrix"}

    assert index["schema"] == SCHEMA
    assert index["status"] == "all_pinned_asymmetric_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == COMPILER
    assert index["rtl_revision"] == RTL_REVISION
    assert index["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert index["mxq_revision"] == MXQ_REVISION
    assert index["source_inventory_sha256"] == _sha(baseline_path)
    assert baseline["rtl_revision"] == RTL_REVISION
    assert len(inventory) == index["matched_sources"] == len(index["cases"]) == 74
    assert index["total_bf16_outputs_checked"] == sum(
        row["compared_bf16_outputs"] for row in index["cases"]) == 360960
    assert {row["source_program"] for row in index["cases"]} == set(inventory)

    assert [group["name"] for group in index["groups"]] == [row[0] for row in GROUPS]
    for group, (_, dim, shape, all_asym, count) in zip(index["groups"], GROUPS):
        matrix = json.loads((ARCHIVE / group["matrix_receipt"]).read_text())
        assert (group["selected_sources"] == matrix["selected_modes"] ==
                matrix["passed_modes"] == count)
        assert group["matrix_receipt_sha256"] == _sha(ARCHIVE / group["matrix_receipt"])
        assert matrix["mesh_dim"] == dim and matrix["source_shape"] == shape
        assert matrix["all_asym_profile"] == (all_asym or dim != 16)
        assert matrix["public_object"] is True
        assert group["matched_bf16_outputs"] == sum(
            row["matched_bf16_outputs"] for row in matrix["rows"])

    for row in index["cases"]:
        receipt_path = ARCHIVE / row["receipt"]
        case = receipt_path.parent
        receipt = json.loads(receipt_path.read_text())
        dispatch_path = case / "object/compile_manifest.json"
        manifest_path = case / "object/object_manifest.json"
        obj_path = case / "object/mx_issue.o"
        physical_path = case / "object/physical_program.json"
        elf_path = case / "physical/asymmetric_program.elf"
        log_path = case / "physical/spike.log"
        dispatch = json.loads(dispatch_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                               f"{row['profile_name']}.json")

        assert row["source_driver_sha256"] == receipt["source_driver_sha256"] == (
            inventory[row["source_program"]])
        assert row["source_header_sha256"] == receipt["source_header_sha256"]
        assert row["receipt_sha256"] == _sha(receipt_path)
        assert row["object_sha256"] == receipt["public_object_sha256"] == _sha(obj_path)
        assert row["elf_sha256"] == receipt["elf_sha256"] == _sha(elf_path)
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"] == _sha(log_path)
        assert row["compared_bf16_outputs"] == receipt["compared_bf16_outputs"]
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
        assert receipt["mxq_revision"] == MXQ_REVISION
        assert receipt["rtl_revision"] == RTL_REVISION
        assert receipt["profile_sha256"] == manifest["profile_sha256"] == profile_sha256(profile)
        assert receipt["public_object_dispatch_sha256"] == _sha(dispatch_path)
        assert receipt["public_object_manifest_sha256"] == _sha(manifest_path)
        assert receipt["physical_program_sha256"] == _sha(physical_path)
        assert dispatch["lowering_family"] == "asymmetric_source"
        assert dispatch["object_sha256"] == manifest["object_sha256"] == row["object_sha256"]
        assert manifest["physical_program_sha256"] == _sha(physical_path)
        assert manifest["allocated_data_section_bytes"] == 0
        assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
        assert any(slot["name"] == "output_bf16" and slot["role"] == "write"
                   for slot in manifest["buffer_abi"])
        assert "0 BF16 mismatches" in log_path.read_text()


def test_six_representative_cases_retain_frontend_and_payload_artifacts() -> None:
    for group, suffix in REPRESENTATIVES:
        path = ARCHIVE / group / suffix
        receipt = json.loads((path / "receipt.json").read_text())
        obj = json.loads((path / "object/object_manifest.json").read_text())
        assert (receipt["generated_c_sha256"] == obj["issuer_c_sha256"] ==
                _sha(path / "object/mx_issue.c") ==
                _sha(path / "physical/mx_issue.c"))
        assert receipt["physical_program_sha256"] == _sha(
            path / "physical/physical_program.json")
        assert (path / "object/physical_program.json").read_bytes() == (
            path / "physical/physical_program.json").read_bytes()
        for name, digest in receipt["files_sha256"].items():
            assert digest == _sha(path / name)
        for name, digest in receipt["physical_inputs_sha256"].items():
            assert digest == _sha(path / "physical" / name)
        for name, digest in obj["resource_sha256"].items():
            assert digest == _sha(path / "physical" / f"{name}.bin")
        driver = (path / "physical/mx_driver.c").read_text()
        assert "mx_issue(" in driver
        assert "gemmini_loop_ws_spad(" not in driver
