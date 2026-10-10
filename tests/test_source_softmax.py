"""Pin the model2MLIR-to-VPU softmax seam and its Spike qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_softmax import (audit_frontend, audit_source,
                                                render_softmax_bound,
                                                replace_source_commands)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_softmax_model2mlir_329718b"
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_softmax_compiler_output_and_spike_reproduction() -> None:
    first = json.loads((EVIDENCE / "first.json").read_text())
    second = json.loads((EVIDENCE / "reproduction.json").read_text())
    assert first == second
    assert first["schema"] == "mx_gemmini.nicolas_vpu_softmax_spike.v1"
    assert first["status"] == "source_vpu_softmax_matched_on_pinned_spike"
    assert first["compiler_revision"] == "329718b2a29bbe9ac1acb38e4425d87f88f60f25"
    assert first["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert first["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert first["spike_exit_code"] == 0
    assert first["compared_bf16_values"] == 512
    assert first["frontend_mlir_sha256"] == _sha(EVIDENCE / "softmax.model2mlir.mlir")
    assert first["bound_mlir_sha256"] == _sha(EVIDENCE / "softmax.profile_bound.mlir")
    assert first["binding_manifest_sha256"] == _sha(EVIDENCE / "binding_manifest.json")
    assert first["files_sha256"]["mx_issue.c"] == _sha(EVIDENCE / "mx_issue.c")
    assert first["spike_log_sha256"] == _sha(EVIDENCE / "spike.log")
    log = (EVIDENCE / "spike.log").read_text()
    assert "softmax 16x32: 0 mismatches vs ref" in log
    assert "vpu_softmax PASSED" in log
    frontend = (EVIDENCE / "softmax.model2mlir.mlir").read_text()
    audit_frontend(frontend)
    mlir = (EVIDENCE / "softmax.profile_bound.mlir").read_text()
    profile = load_profile(PROFILE)
    assert first["profile_sha256"] == profile_sha256(profile)
    commands = lower_vector_commands(mlir, profile)
    assert len(commands) == 6
    assert [command.funct for command in commands] == [33] * 6
    assert [command.rs2.immediate & 0xf for command in commands] == [8, 1, 5, 9, 6, 2]
    assert (EVIDENCE / "mx_issue.c").read_text().count(".insn r 0x7b, 3, 33") == 6


def test_softmax_binding_refuses_changed_pinned_source() -> None:
    rtl = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
    source_path = rtl / "software/gemmini-rocc-tests/bareMetalC/vpu_softmax.c"
    if not source_path.is_file():
        pytest.skip("Nicolas's pinned source checkout is not installed")
    source = source_path.read_text()
    assert _sha(source_path) == json.loads((EVIDENCE / "first.json").read_text())[
        "source_sha256"]
    audit_source(source)
    transformed = replace_source_commands(source)
    assert "gemmini_vpu_" not in transformed
    assert transformed.count("mx_issue();") == 1
    frontend = (EVIDENCE / "softmax.model2mlir.mlir").read_text()
    profile = load_profile(PROFILE, rtl_root=rtl)
    mlir, manifest = render_softmax_bound(
        frontend, source, profile, CONTRACT.read_bytes())
    assert mlir == (EVIDENCE / "softmax.profile_bound.mlir").read_text()
    assert manifest == json.loads((EVIDENCE / "binding_manifest.json").read_text())
    with pytest.raises(ValueError, match="schedule"):
        audit_source(source.replace("VPU_RMAX", "VPU_RSUM", 1))
    with pytest.raises(ValueError, match="capture"):
        audit_frontend(frontend.replace("math.exp", "math.expm1", 1))
