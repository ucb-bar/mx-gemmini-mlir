"""The source attention VPU middle must follow the captured graph and profile."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.nicolas_attention_vector import _audit_frontend_edges
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vector_lowering import lower_vector_commands
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
COMPILED = ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_typed_attention_vector_matches_source_and_spike_stage_hashes() -> None:
    bound = (COMPILED / "attention.vector_bound.mlir").read_text()
    binding = json.loads((COMPILED / "vector_binding.json").read_text())
    receipt = json.loads((COMPILED / "receipt.json").read_text())
    source = json.loads((SOURCE / "receipt.json").read_text())
    report = verify_ir(bound, PROFILE)
    commands = lower_vector_commands(bound, PROFILE)
    physical = json.loads((COMPILED / "physical_commands.json").read_text())
    assert (report["vpu_commands"], report["spad_requants"]) == (7, 1)
    assert [command.funct for command in commands] == [33] * 7 + [34]
    assert physical["commands"] == [asdict(command) for command in commands]
    assert receipt["physical_commands_sha256"] == _sha(COMPILED / "physical_commands.json")
    assert commands[0].rs2.immediate & 0xf == 4  # scalar multiply
    assert commands[0].rs2.immediate >> 16 == 0x3e00  # BF16 1/8
    assert commands[-1].rs1.buffer == "p_scales"
    assert commands[-1].rs1.or_bits & 0x3fff == 384  # score scratchpad row
    assert (commands[-1].rs1.or_bits >> 14) & 0x3fff == 640  # resident P row
    assert binding["source_sha256"] == source["npu_source_sha256"]
    assert binding["frontend_mlir_sha256"] == source["frontend_mlir_sha256"]
    assert binding["source_header_sha256"] == source["generated_header_sha256"]
    assert binding["bound_mlir_sha256"] == _sha(COMPILED / "attention.vector_bound.mlir")
    assert receipt["bound_mlir_sha256"] == binding["bound_mlir_sha256"]
    assert receipt["issuer_c_sha256"] == _sha(COMPILED / "mx_issue.c")
    assert receipt["object_sha256"] == _sha(COMPILED / "mx_issue.o")
    assert receipt["object_data_section_bytes"] == 0
    assert receipt["source_stage_hashes"] == source["source_stage_hashes"]
    driver = (COMPILED / "attn_vpu_generated_middle.c").read_text()
    assert driver.count("mx_issue(p_scales);") == 1
    assert "gemmini_vpu_" not in driver and "gemmini_spad_requant(" not in driver
    assert _sha(COMPILED / "attn_vpu_generated_middle.c") == receipt["driver_sha256"]
    assert "check hashes vs Spike: match (0 of 4 differ)" in (
        COMPILED / "generated_middle.log").read_text()
    assert "attn_vpu PASSED" in (COMPILED / "generated_middle.log").read_text()
    assert receipt["spike_log_sha256"] == _sha(COMPILED / "generated_middle.log")
    assert receipt["scope"].endswith("QK and PV matrix issue remains in source C")


def test_frontend_rejects_changed_scale_or_broken_qk_edge() -> None:
    frontend = (SOURCE / "attention.model2mlir.mlir").read_text()
    _audit_frontend_edges(frontend)
    with pytest.raises(ValueError, match="source scale"):
        _audit_frontend_edges(frontend.replace("1.250000e-01 : f32",
                                               "2.500000e-01 : f32", 1))
    with pytest.raises(ValueError, match="SSA edges"):
        _audit_frontend_edges(frontend.replace("ins(%5, %12 :", "ins(%0, %12 :", 1))


def test_bound_attention_vector_rejects_other_profile() -> None:
    bound = (COMPILED / "attention.vector_bound.mlir").read_text()
    assert 'site_id = "softmax:6"' in bound
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="profile digest differs"):
        lower_vector_commands(bound, profile)
