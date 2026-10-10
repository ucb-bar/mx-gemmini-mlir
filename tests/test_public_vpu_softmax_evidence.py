"""Keep the public-object Spike qualification for both Nicolas VPU profiles auditable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/mx_public_vpu_softmax_3f9af55"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
COMPILER = "3f9af556cb6b25046f0ba896c8db6a1576723d41"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("folder", "profile_name"), [
    ("e4m3_fp4_vpu", "MxE4M3Fp4VpuGemminiRocketConfig"),
    ("e4m3_vpu", "MxE4M3VpuGemminiRocketConfig"),
])
def test_public_object_softmax_matches_all_bf16_outputs(folder: str,
                                                         profile_name: str) -> None:
    path = EVIDENCE / folder
    receipt = json.loads((path / "artifact_manifest.json").read_text())
    obj = json.loads((path / "object/object_manifest.json").read_text())
    dispatch = json.loads((path / "object/compile_manifest.json").read_text())
    profile = load_profile(PROFILES / f"{profile_name}.json")
    assert receipt["status"] == "source_vpu_softmax_matched_on_pinned_spike"
    assert receipt["spike_exit_code"] == 0
    assert receipt["compared_bf16_values"] == 512
    assert receipt["compiler_revision"] == obj["compiler_revision"] == COMPILER
    assert receipt["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert receipt["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert receipt["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
    assert receipt["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == _sha(
        path / "softmax.profile_bound.mlir")
    assert receipt["frontend_mlir_sha256"] == _sha(path / "softmax.model2mlir.mlir")
    assert receipt["binding_manifest_sha256"] == _sha(path / "binding_manifest.json")
    assert receipt["object_compile_manifest_sha256"] == _sha(
        path / "object/compile_manifest.json")
    assert receipt["object_manifest_sha256"] == dispatch["object_manifest_sha256"] == _sha(
        path / "object/object_manifest.json")
    assert receipt["spike_log_sha256"] == _sha(path / "spike.log")
    assert obj["object_sha256"] == dispatch["object_sha256"] == _sha(
        path / "object/mx_issue.o")
    assert obj["issuer_c_sha256"] == receipt["files_sha256"]["mx_issue.c"] == _sha(
        path / "object/mx_issue.c")
    assert obj["allocated_data_section_bytes"] == obj["embedded_operand_bytes"] == 0
    assert obj["embedded_golden_bytes"] == 0
    assert dispatch["lowering_family"] == "vpu_softmax"
    assert obj["command_count"] == 17
    assert [entry["slot"] for entry in obj["buffer_abi"]] == ["score", "output"]
    physical = json.loads((path / "object/physical_program.json").read_text())
    assert [command["funct"] for command in physical["commands"]] == [
        7, 0, 0, *([2] * 4), *([33] * 6), *([3] * 4)]
    log = (path / "spike.log").read_text()
    assert "softmax 16x32: 0 mismatches vs ref" in log
    assert "vpu_softmax PASSED" in log
