"""Audit Nicolas's compiler-generated 128x128x512 FP4 requantized replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify
from tools.qualify_nicolas_plain_matrix_object import CASES, MODEL2MLIR_REVISION, RTL_REVISION


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp4_large_requant_typed_object_03490a5_266c593"
CASE = CASES["fp4_128x128x512_requant"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nicolas_large_fp4_requant_matches_source_packed_bytes_and_scales() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    obj = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    dispatch = json.loads((EVIDENCE / "object/compile_manifest.json").read_text())
    physical = json.loads((EVIDENCE / "object/physical_program.json").read_text())
    profile = load_profile(ROOT / f"profiles/gemmini-mx-cleanup-266c593/{CASE.profile_name}.json")
    bundle, resources = load_bundle(EVIDENCE / "bundle")
    family, report = classify((EVIDENCE / "payload_bound.mlir").read_text(), profile)

    assert receipt["schema"] == "mx_gemmini.nicolas_plain_fp4_typed_object_spike.v1"
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["mismatches"] == 0
    assert receipt["case"] == CASE.key
    assert receipt["packed_bytes_checked"] == 8192
    assert receipt["scales_checked"] == 512
    assert receipt["outputs_checked"] == 8704
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["compiler_revision"] == dispatch["compiler_revision"] == (
        "03490a58996fd145fc6bb0aa6b08f9702b40ac62")
    assert receipt["source_driver_sha256"] == bundle["source_driver_sha256"] == CASE.source_sha256
    assert receipt["source_header_sha256"] == bundle["source_header_sha256"] == CASE.header_sha256
    assert receipt["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
    assert bundle["origin"] == "nicolas_source_header_specialization"
    assert bundle["output_format"] == "fp4_e2m1"
    assert bundle["source_quant_header_format"] == "packed_fp4_e2m1"
    assert bundle["shape_mnk"] == [128, 128, 512]
    assert resources["source_fp4_packed"] == resources["nicolas_fp4"]
    assert resources["golden_output_scales"] == resources["nicolas_output_scales"]
    assert len(resources["source_fp4_packed"]) == 8192
    assert len(resources["golden_output_scales"]) == 512
    assert family == dispatch["lowering_family"] == "source_contract"
    assert report["contracts"] == 1 and report["source_resources"] == 4
    assert physical["output_format"] == "fp4_e2m1"
    assert physical["plan"]["quant_output"] is True
    assert [slot["name"] for slot in obj["buffer_abi"]] == list(CASE.buffer_abi)
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0

    for key, path in (
            ("source_mlir_sha256", "model2mlir.mlir"),
            ("handoff_mlir_sha256", "handoff.mlir"),
            ("payload_manifest_sha256", "bundle/manifest.json"),
            ("bound_mlir_sha256", "payload_bound.mlir"),
            ("object_dispatch_manifest_sha256", "object/compile_manifest.json"),
            ("object_manifest_sha256", "object/object_manifest.json"),
            ("object_sha256", "object/mx_issue.o"),
            ("elf_sha256", "run/mx_program.elf"),
            ("spike_log_sha256", "run/spike.log"),
            ("driver_sha256", "run/mx_driver.c")):
        assert receipt[key] == _sha(EVIDENCE / path)
    assert obj["object_sha256"] == dispatch["object_sha256"]
    assert obj["physical_program_sha256"] == _sha(EVIDENCE / "object/physical_program.json")
    assert "compiled Nicolas FP4 128x128x512 requant: 0 packed-byte mismatches / 8192, " \
           "0 scale mismatches / 512" in (EVIDENCE / "run/spike.log").read_text()
    driver = (EVIDENCE / "run/mx_driver.c").read_text()
    assert "C_out[i][j]" in driver and "C_scales_out[i][g]" in driver
    assert "gemmini_loop_ws_spad" not in driver
