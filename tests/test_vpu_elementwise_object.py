"""Check public VPU command lowering against Nicolas's captured VPU cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_elementwise_program import lower_vpu_elementwise_program
from tools.compile_object import classify
from tools.qualify_nicolas_vpu_elementwise import KINDS, REDUCING, bind as bind_base
from tools.qualify_nicolas_vpu_fused import bind as bind_fused


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
ARCHIVE = ROOT / "docs/evidence/nicolas_vpu_elementwise_compiled_266c593"


@pytest.mark.parametrize("kind", KINDS)
def test_public_lowerer_reproduces_nicolas_base_vpu_issuer(kind: str) -> None:
    bound = (ARCHIVE / kind / "bound.mlir").read_text()
    assert classify(bound, PROFILE)[0] == "vpu_elementwise"
    buffers = {"src1": "src1", "src2": "src2", "output": "output"}
    plan = lower_vpu_elementwise_program(bound, PROFILE, buffers)
    issuer = emit_c(list(plan.commands), transport="rocket_rocc",
                    buffers=("src1", "src2", "output"))
    assert issuer == (ARCHIVE / kind / "mx_issue.c").read_text()
    assert plan.kind == kind
    assert plan.rows == 64
    assert plan.output_rows == (16 if kind in REDUCING else 64)
    assert [command.funct for command in plan.commands].count(33) == 1


def test_public_lowerer_exposes_expsum_second_output() -> None:
    bound = (ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593/"
             "expsum/bound.mlir").read_text()
    plan = lower_vpu_elementwise_program(
        bound, PROFILE, {"src1": "a", "src2": "b",
                         "output": "output", "output2": "sums"})
    assert plan.kind == "expsum"
    assert plan.src2_rows == 16
    assert plan.output_rows == 64
    assert plan.second_dst_row == 12288
    assert [command.funct for command in plan.commands].count(3) == 5
    with pytest.raises(ValueError, match="buffer map"):
        lower_vpu_elementwise_program(
            bound, PROFILE, {"src1": "a", "src2": "b", "output": "output"})


def test_public_lowerer_broadcast_expsub_uses_one_source_group() -> None:
    bound = (ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593/"
             "expsub/bound.mlir").read_text()
    plan = lower_vpu_elementwise_program(
        bound, PROFILE, {"src1": "a", "src2": "b", "output": "output"})
    assert plan.kind == "expsub"
    assert plan.src2_rows == 16
    assert plan.output_rows == 64
    assert plan.second_dst_row is None


def test_public_lowerer_rejects_profile_without_vpu() -> None:
    plain = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    bound = (ARCHIVE / "add/bound.mlir").read_text()
    with pytest.raises(ValueError, match="profile digest|no VPU"):
        lower_vpu_elementwise_program(
            bound, plain, {"src1": "a", "src2": "b", "output": "output"})


@pytest.mark.parametrize("kind", (*KINDS, "expsub", "expsum"))
def test_vpu_capture_rebinds_to_fp4_capable_profile_without_command_change(
        kind: str) -> None:
    fp4_vpu = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/"
        "MxE4M3Fp4VpuGemminiRocketConfig.json")
    fused = kind in {"expsub", "expsum"}
    archived = (ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593"
                if fused else ARCHIVE) / kind
    # The binding carries the pinned source hash; use the archived value so
    # this test also verifies that the original bound module is reproducible.
    source_sha = json.loads((archived / "binding.json").read_text())["source_sha256"]
    original, _ = (bind_fused if fused else bind_base)(
        kind, (archived / "frontend.mlir").read_text(), source_sha, PROFILE)
    rebound, _ = (bind_fused if fused else bind_base)(
        kind, (archived / "frontend.mlir").read_text(), source_sha, fp4_vpu)
    assert original == (archived / "bound.mlir").read_text()
    assert classify(rebound, fp4_vpu)[0] == "vpu_elementwise"
    buffers = {"src1": "a", "src2": "b", "output": "output"}
    if kind == "expsum":
        buffers["output2"] = "sums"
    assert (lower_vpu_elementwise_program(original, PROFILE, buffers).commands ==
            lower_vpu_elementwise_program(rebound, fp4_vpu, buffers).commands)
