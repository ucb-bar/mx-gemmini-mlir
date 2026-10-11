"""A dependent VPU chain must keep its sources, writes, and command order."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_sequence_program import lower_vpu_sequence
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
ABI = {
    "schema": "mx_gemmini.vpu_sequence_buffer_map.v1",
    "inputs": [{"name": "a", "row": 0, "rows": 64},
               {"name": "b", "row": 4096, "rows": 64}],
    "outputs": [{"name": "middle", "row": 8192, "rows": 64},
                {"name": "result", "row": 12288, "rows": 16}],
}


def _graph() -> str:
    profile = profile_sha256(PROFILE)
    contract, policy, manifest = ("a" * 64, "b" * 64, "c" * 64)
    operations = (("add", 0, 4096, 8192, 1, 0),
                  ("muls", 8192, 0, 8192, 1, 0x3f00),
                  ("rmax", 8192, 0, 12288, 4, 0))
    body = "\n".join(
        f'    "mx_gemmini.vpu_execute"() {{site_id = "chain:{i}", '
        f'kind = "{kind}", src1_row = {src1} : i32, '
        f'src2_row = {src2} : i32, dst_row = {dst} : i32, '
        f'rows = 64 : i32, reduction_length = {rlen} : i32, '
        f'broadcast = false, immediate_bf16 = {imm} : i32, '
        f'profile_sha256 = "{profile}", contract_sha256 = "{contract}", '
        f'policy_sha256 = "{policy}", manifest_sha256 = "{manifest}"}} : () -> ()'
        for i, (kind, src1, src2, dst, rlen, imm) in enumerate(operations))
    return (
        f'builtin.module attributes {{mx.profile_sha256 = "{profile}", '
        f'mx.contract_sha256 = "{contract}", mx.policy_sha256 = "{policy}", '
        f'prov.quantization_manifest_sha256 = "{manifest}", '
        'mx.vpu_sequence_schema = "mx_gemmini.vpu_sequence.v1"} {\n'
        '  func.func @chain() {\n' + body + '\n    func.return\n  }\n}\n')


def test_dependent_vpu_sequence_lowers_in_order() -> None:
    mlir = _graph()
    family, report = classify(mlir, PROFILE)
    assert family == "vpu_sequence"
    assert report["vpu_commands"] == 3
    plan = lower_vpu_sequence(mlir, PROFILE, ABI)
    assert [op["kind"] for op in plan.operations] == ["add", "muls", "rmax"]
    assert [command.funct for command in plan.commands].count(33) == 3
    assert plan.operations[-1]["dst_rows"] == 16
    assert [item.name for item in plan.outputs] == ["middle", "result"]


def test_sequence_rejects_uninitialized_operand_and_output() -> None:
    mlir = _graph()
    with pytest.raises(ValueError, match="uninitialized source 1"):
        lower_vpu_sequence(mlir.replace("src1_row = 8192 : i32",
                                        "src1_row = 8000 : i32", 1), PROFILE, ABI)
    altered = {**ABI, "outputs": [
        {"name": "result", "row": 12000, "rows": 16}]}
    with pytest.raises(ValueError, match="uninitialized"):
        lower_vpu_sequence(mlir, PROFILE, altered)
    missing_b = {**ABI, "inputs": [ABI["inputs"][0]]}
    with pytest.raises(ValueError, match="uninitialized source 2"):
        lower_vpu_sequence(mlir, PROFILE, missing_b)
    overlapping = {**ABI, "inputs": [
        ABI["inputs"][0], {"name": "overlap", "row": 63, "rows": 2}]}
    with pytest.raises(ValueError, match="spans overlap"):
        lower_vpu_sequence(mlir, PROFILE, overlapping)


def test_sequence_requires_explicit_schema_and_flat_function() -> None:
    mlir = _graph()
    with pytest.raises(ValueError, match="unsupported"):
        classify(mlir.replace("mx_gemmini.vpu_sequence.v1",
                              "mx_gemmini.vpu_sequence.v2"), PROFILE)
    with pytest.raises(ValueError, match="ordered physical VPU"):
        classify(mlir.replace("    func.return",
                              '    %c0 = "arith.constant"() {value = 0 : i32} : () -> i32\n'
                              "    func.return"), PROFILE)
