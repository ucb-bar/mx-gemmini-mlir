"""Audit the expanded direct Nicolas source matrix suite and its new objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)
from tools.qualify_nicolas_plain_matrix_suite import SCHEMA


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_direct_matrix_suite_9df3384_266c593"
COMPILER = "9df3384937e89b4b79f5234d68379fdf5a5388ef"
NEW_CASES = {
    "fp8_32x32x32", "fp8_64x256x64_requant_dim32",
    "fp8_128x128x256_requant_dim32", "fp4_128x128x512",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_expanded_suite_replays_all_registered_direct_cases() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    assert index["schema"] == SCHEMA
    assert index["status"] == "all_selected_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == COMPILER
    assert index["rtl_revision"] == RTL_REVISION
    assert index["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert index["mxq_revision"] == MXQ_REVISION
    assert len(index["selected_cases"]) == len(set(index["selected_cases"])) == 15
    assert set(index["selected_cases"]).issubset(CASES)
    assert NEW_CASES.issubset(index["selected_cases"])
    assert [row["case"] for row in index["cases"]] == index["selected_cases"]
    assert index["total_outputs_checked"] == sum(
        row["outputs_checked"] for row in index["cases"]) == 163072

    for row in index["cases"]:
        case = CASES[row["case"]]
        receipt_path = ARCHIVE / row["receipt"]
        receipt = json.loads(receipt_path.read_text())
        m, n, _ = case.shape
        outputs = (m * n // (2 if case.precision in {"FP4", "FP6"} else 1) +
                   m * n // 32 if case.quant_output else m * n)
        assert row["receipt_sha256"] == _sha(receipt_path)
        assert row["outputs_checked"] == receipt["outputs_checked"] == outputs
        assert row["precision"] == case.precision
        assert row["shape_mnk"] == list(case.shape)
        assert row["source_driver_sha256"] == receipt["source_driver_sha256"] == case.source_sha256
        assert row["source_header_sha256"] == receipt["source_header_sha256"] == case.header_sha256
        assert row["profile_name"] == receipt["profile_name"] == case.profile_name
        assert row["object_sha256"] == receipt["object_sha256"]
        assert row["elf_sha256"] == receipt["elf_sha256"]
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"]
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["mismatches"] == 0


def test_new_shapes_have_reviewable_frontend_object_and_full_output_artifacts() -> None:
    for key in NEW_CASES:
        case = CASES[key]
        path = ARCHIVE / key
        receipt = json.loads((path / "receipt.json").read_text())
        dispatch = json.loads((path / "object/compile_manifest.json").read_text())
        obj = json.loads((path / "object/object_manifest.json").read_text())
        physical = json.loads((path / "object/physical_program.json").read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                               f"{case.profile_name}.json")
        bundle, resources = load_bundle(path / "bundle")
        family, report = classify((path / "payload_bound.mlir").read_text(), profile)

        assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
        assert receipt["mxq_revision"] == MXQ_REVISION
        assert receipt["rtl_revision"] == RTL_REVISION
        assert (receipt["profile_sha256"] == obj["profile_sha256"] ==
                dispatch["profile_sha256"] == profile_sha256(profile))
        assert family == dispatch["lowering_family"] == "source_contract"
        assert report["contracts"] == 1
        assert bundle["source_driver_sha256"] == case.source_sha256
        assert bundle["source_header_sha256"] == case.header_sha256
        assert tuple(bundle["shape_mnk"]) == case.shape
        assert tuple(bundle["tile_mnk"]) == case.tile
        assert len(resources["golden_bf16"]) == case.shape[0] * case.shape[1] * 2
        assert physical["plan"]["shape"] == list(case.shape)
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert [slot["name"] for slot in obj["buffer_abi"]] == list(case.buffer_abi)

        for field, relative in (
                ("source_mlir_sha256", "model2mlir.mlir"),
                ("handoff_mlir_sha256", "handoff.mlir"),
                ("payload_manifest_sha256", "bundle/manifest.json"),
                ("bound_mlir_sha256", "payload_bound.mlir"),
                ("object_sha256", "object/mx_issue.o"),
                ("object_dispatch_manifest_sha256", "object/compile_manifest.json"),
                ("object_manifest_sha256", "object/object_manifest.json"),
                ("elf_sha256", "run/mx_program.elf"),
                ("spike_log_sha256", "run/spike.log"),
                ("driver_sha256", "run/mx_driver.c")):
            assert receipt[field] == _sha(path / relative)
        assert obj["object_sha256"] == dispatch["object_sha256"] == receipt["object_sha256"]
        log = (path / "run/spike.log").read_text()
        assert f"compiled Nicolas {case.label}: 0 " in log
        if case.quant_output:
            assert f"0 scale mismatches / {case.shape[0] * case.shape[1] // 32}" in log
        else:
            assert f"0 mismatches / {case.shape[0] * case.shape[1]} BF16 values" in log
        driver = (path / "run/mx_driver.c").read_text()
        assert "mx_issue(" in driver
        assert "gemmini_loop_ws_spad(" not in driver
