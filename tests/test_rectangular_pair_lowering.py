"""Physical layout gates for a connected pair with distinct contraction shapes."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.resident_lowering import lower_resident_contract
from mx_gemmini_support.resident_pair_plan import plan_fp8_rectangular_pair
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def test_rectangular_pair_places_b1_and_b2_from_their_own_shapes() -> None:
    profile = load_profile(PROFILE)
    plan = plan_fp8_rectangular_pair(
        profile, first_shape=(64, 96, 64), second_shape=(64, 32, 96),
        a_row=0, c1_row=2048, c2_row=4096)
    assert (plan.a_rows, plan.b1_rows, plan.b_rows,
            plan.c1_rows, plan.c_rows) == (256, 384, 192, 384, 128)
    assert (plan.b1_row, plan.b_row) == (16000, 16192)
    with pytest.raises(ValueError, match="SSA shapes"):
        plan_fp8_rectangular_pair(
            profile, first_shape=(64, 96, 64), second_shape=(64, 32, 64),
            a_row=0, c1_row=2048, c2_row=4096)
    with pytest.raises(ValueError, match="lifetimes overlap"):
        plan_fp8_rectangular_pair(
            profile, first_shape=(64, 96, 64), second_shape=(64, 32, 96),
            a_row=0, c1_row=128, c2_row=4096)


def test_rectangular_second_weight_load_uses_k_major_tile_order() -> None:
    profile = load_profile(PROFILE)
    attrs = {
        "m": 64, "n": 64, "k": 96,
        "activation_row": 2048, "weight_row": 16000, "output_row": 4096,
        "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
        "output_format": "fp8_e4m3", "weight_buffer": "b2_weight",
        "weight_scales_buffer": "b2_scales", "output_scales_buffer": "c2_scales",
    }
    commands = lower_resident_contract(profile, attrs)
    transfers = [command for command in commands if isinstance(command, Command)
                 and command.funct == 2]
    assert [(command.rs1.byte_offset, command.rs2.immediate & 0xffffffff)
            for command in transfers] == [
                (ki * 16 * 64 + nj * 16, 16000 + (ki * 4 + nj) * 16)
                for ki in range(6) for nj in range(4)]
