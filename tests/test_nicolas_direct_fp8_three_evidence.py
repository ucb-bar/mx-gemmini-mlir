"""Audit three additional pinned Nicolas FP8 public-object replays."""

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
ARCHIVE = ROOT / "docs/evidence/nicolas_direct_fp8_three_789d192_266c593"
COMPILER = "789d192aadc3ad6efb65ad4c15bf310000fb47ed"
NEW_CASES = ("fp8_64x64x64", "fp8_64x64x64_requant", "fp8_96x32x32")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_fp8_sources_have_full_public_object_spike_replays() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hashes = {entry["name"]: entry["source_sha256"]
                     for entry in inventory["entries"]}
    assert index["schema"] == SCHEMA
    assert index["status"] == "all_selected_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == COMPILER
    assert index["rtl_revision"] == RTL_REVISION
    assert index["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert index["mxq_revision"] == MXQ_REVISION
    assert index["selected_cases"] == list(NEW_CASES)
    assert [row["case"] for row in index["cases"]] == list(NEW_CASES)
    assert index["total_outputs_checked"] == 11392

    for row in index["cases"]:
        case = CASES[row["case"]]
        path = ARCHIVE / case.key
        receipt = json.loads((path / "receipt.json").read_text())
        dispatch = json.loads((path / "object/compile_manifest.json").read_text())
        obj = json.loads((path / "object/object_manifest.json").read_text())
        physical = json.loads((path / "object/physical_program.json").read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                               f"{case.profile_name}.json")
        bundle, resources = load_bundle(path / "bundle")
        family, report = classify((path / "payload_bound.mlir").read_text(), profile)
        m, n, _ = case.shape
        output_count = m * n + (m * n // 32 if case.quant_output else 0)

        assert case.source_sha256 == source_hashes[case.source_name.removesuffix(".c")]
        assert row["source_driver_sha256"] == receipt["source_driver_sha256"] == case.source_sha256
        assert row["source_header_sha256"] == receipt["source_header_sha256"] == case.header_sha256
        assert row["receipt_sha256"] == _sha(path / "receipt.json")
        assert row["outputs_checked"] == receipt["outputs_checked"] == output_count
        assert row["object_sha256"] == receipt["object_sha256"] == _sha(path / "object/mx_issue.o")
        assert row["elf_sha256"] == receipt["elf_sha256"] == _sha(path / "run/mx_program.elf")
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"] == _sha(path / "run/spike.log")
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["mismatches"] == 0
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
        assert family == dispatch["lowering_family"] == "source_contract"
        assert report["contracts"] == 1
        assert bundle["source_driver_sha256"] == case.source_sha256
        assert bundle["source_header_sha256"] == case.header_sha256
        assert tuple(bundle["shape_mnk"]) == case.shape
        assert len(resources["golden_bf16"]) == m * n * 2
        assert physical["plan"]["shape"] == list(case.shape)
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert [slot["name"] for slot in obj["buffer_abi"]] == list(case.buffer_abi)
        for field, relative in (
                ("source_mlir_sha256", "model2mlir.mlir"),
                ("handoff_mlir_sha256", "handoff.mlir"),
                ("payload_manifest_sha256", "bundle/manifest.json"),
                ("bound_mlir_sha256", "payload_bound.mlir"),
                ("object_dispatch_manifest_sha256", "object/compile_manifest.json"),
                ("object_manifest_sha256", "object/object_manifest.json"),
                ("driver_sha256", "run/mx_driver.c")):
            assert receipt[field] == _sha(path / relative)
        assert "mx_issue(" in (path / "run/mx_driver.c").read_text()
        assert "gemmini_loop_ws_spad(" not in (path / "run/mx_driver.c").read_text()
        log = (path / "run/spike.log").read_text()
        if case.quant_output:
            assert f"0 code mismatches / {m * n}, 0 scale mismatches / {m * n // 32}" in log
        else:
            assert f"0 mismatches / {m * n} BF16 values" in log
