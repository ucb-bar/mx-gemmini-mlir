"""MX attention transfer packets retain Nicolas's scratchpad placement."""

from pathlib import Path

import pytest

from mx_gemmini_support.nicolas_attention_io import (
    lower_dense_attention_operands, lower_dense_attention_readout,
    lower_dense_attention_scales)
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_attn_compute import _driver, _driver_with_io


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def test_scale_operand_and_readout_packets() -> None:
    profile = load_profile(PROFILE / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    scales = lower_dense_attention_scales(profile)
    assert [item.funct for item in scales] == [27] * 3
    assert scales[0].rs1.buffer == "Q_SCALES" and scales[0].rs2.immediate == 64
    assert scales[1].rs1.buffer == "KT_SCALES" and scales[1].rs2.immediate == (1 << 32) | 128
    assert scales[2].rs1.buffer == "V_SCALES"
    assert scales[2].rs1.or_bits == 64 << 40
    assert scales[2].rs2.immediate == (2 << 46) | (4096 << 33) | (1 << 32) | 64
    operands = lower_dense_attention_operands(profile)
    assert len(operands) == 43
    assert [item.funct for item in operands].count(2) == 40
    assert [item.funct for item in operands].count(0) == 3
    assert operands[1].rs1.buffer == "Q_IN"
    assert operands[10].rs1.buffer == "KT_IN"
    assert operands[27].rs1.buffer == "V_IN"
    for stage, buffer, start, rows in (
            ("score", "S_hw", 384, 256),
            ("probability", "P_hw", 384, 256),
            ("probability_codes", "p_codes_hw", 640, 128),
            ("output", "O_hw", 1024, 256)):
        commands = lower_dense_attention_readout(profile, stage)
        assert len(commands) == 1 + rows // 16
        assert commands[0].funct == 0 and commands[0].rs2.immediate == 16
        assert commands[1].funct == 3 and commands[1].rs1.buffer == buffer
        assert commands[1].rs2.immediate & 0xffffffff == start
    with pytest.raises(ValueError, match="unsupported"):
        lower_dense_attention_readout(profile, "mask")


def test_attention_io_rejects_plain_mx_profile() -> None:
    plain = load_profile(PROFILE / "MxGemminiRocketConfig.json")
    with pytest.raises(ValueError, match=r"MX\+VPU"):
        lower_dense_attention_scales(plain)


def test_generated_io_driver_removes_source_transfer_calls() -> None:
    middle = (ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb/"
              "attn_vpu_generated_middle.c").read_text()
    compute = _driver(middle)
    driver = _driver_with_io(compute)
    for name in ("scales", "operands", "score", "probability",
                 "probability_codes", "output"):
        assert f"void mx_issue_{name}(" in driver
        assert f"mx_issue_{name}(" in driver
    for source_call in ("mvin_A(", "mvin_B(", "mvout_rows(",
                        "gemmini_mx_load_scales(", "gemmini_mx_load_scales_2d("):
        assert source_call not in driver
    with pytest.raises(ValueError, match="source transfer or readout seam differs"):
        _driver_with_io(compute.replace("sizeof(KT_SCALES)", "64", 1))
