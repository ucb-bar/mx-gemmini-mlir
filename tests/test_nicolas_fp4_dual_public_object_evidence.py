"""Audit the public-object replay of Nicolas's dual-layout FP4 requantizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp4_dual_public_object_d4ed0d8_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
COMPILER = "d4ed0d8e0fea4acb089a198f7fec11dbd03f67f1"
RTL = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_fp4_dual_object_replays_entire_source_oracle() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    dispatch = json.loads((EVIDENCE / "object/compile_manifest.json").read_text())
    obj = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    physical = json.loads((EVIDENCE / "object/physical_program.json").read_text())
    mlir = (EVIDENCE / "connected.mlir").read_text()
    profile = load_profile(PROFILE)
    family, report = classify(mlir, profile)

    assert receipt["status"] == "source_and_public_object_matched_on_pinned_spike"
    assert receipt["schema"] == "mx_gemmini.nicolas_fp4_dual_public_object_spike.v1"
    assert (receipt["compiler_revision"] == dispatch["compiler_revision"] ==
            obj["compiler_revision"] == COMPILER)
    assert receipt["rtl_revision"] == obj["rtl_revision"] == RTL
    assert (receipt["profile_sha256"] == obj["profile_sha256"] ==
            dispatch["profile_sha256"] == profile_sha256(profile))
    assert family == dispatch["lowering_family"] == "fp4_dual_requant"
    assert report["spad_requants"] == 2
    assert obj["schema"] == "mx_gemmini.fp4_dual_requant_linkable_object.v1"
    assert obj["command_count"] == receipt["command_count"] == 101
    assert obj["fence_count"] == 1
    assert receipt["allocated_data_section_bytes"] == obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert [(slot["name"], slot["minimum_bytes"]) for slot in obj["buffer_abi"]] == [
        ("X", 16384), ("scales_hw", 256), ("scales_hw2", 256),
        ("codes_flat_hw", 4096), ("codes_tiled_hw", 4096)]
    assert sum(command.get("funct") == 34 for command in physical["commands"]) == 2
    assert (physical["bound_mlir_sha256"] == obj["bound_mlir_sha256"] ==
            dispatch["bound_mlir_sha256"] == _sha(EVIDENCE / "connected.mlir"))

    for field, relative in (
            ("connected_mlir_sha256", "connected.mlir"),
            ("driver_sha256", "compiler_driver.c"),
            ("driver_patch_sha256", "compiler_driver.patch"),
            ("object_dispatch_manifest_sha256", "object/compile_manifest.json"),
            ("object_manifest_sha256", "object/object_manifest.json"),
            ("physical_program_sha256", "object/physical_program.json"),
            ("issuer_c_sha256", "object/mx_issue.c"),
            ("object_sha256", "object/mx_issue.o")):
        assert receipt[field] == _sha(EVIDENCE / relative)
    assert obj["object_sha256"] == dispatch["object_sha256"] == receipt["object_sha256"]
    assert receipt["object_sha256"] == (
        "ab3975ad7ad33a913ceadc7858d3ab934ab1b41a5aa16f54bb9be8b733e7b067")
    assert dispatch["object_manifest_sha256"] == receipt["object_manifest_sha256"]

    for side in ("source", "compiler"):
        result = receipt[f"{side}_spike"]
        folder = "compiled" if side == "compiler" else "source"
        assert result["exit_code"] == 0
        assert result["compared_fp4_codes"] == 16384
        assert result["compared_e8m0_scales"] == 512
        assert result["elf_sha256"] == _sha(EVIDENCE / folder / "program.elf")
        assert result["spike_log_sha256"] == _sha(EVIDENCE / folder / "spike.log")
        log = (EVIDENCE / folder / "spike.log").read_text()
        assert "flat 0, tiled 0 code mismatches; scales 0, 0 mismatches" in log
        assert "spad_requant_fp4 PASSED" in log

    driver = (EVIDENCE / "compiler_driver.c").read_text()
    assert "mx_issue(X, scales_hw, scales_hw2, codes_flat_hw, codes_tiled_hw);" in driver
    assert "gemmini_spad_requant_fp4(" not in driver
    assert "gemmini_extended_mvin(" not in driver
    assert "gemmini_extended_mvout(" not in driver
    assert "mxr_scale(&X[m][32 * b], 32)" in driver
