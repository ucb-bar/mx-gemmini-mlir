"""Profile-checked transfer phases used by Nicolas's MX memory benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mx_gemmini_support.memory_phase import (
    lower_linear_spad_mvout, lower_matrix_mvin, lower_scale_load,
)
from mx_gemmini_support.target_profile import load_profile


PROFILE = Path(__file__).resolve().parents[1] / (
    "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")


def test_nicolas_memory_phase_command_geometry() -> None:
    profile = load_profile(PROFILE)
    a = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=0)
    b = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=16, spad_row=1024)
    scales = lower_scale_load(profile, buffer="src", payload_bytes=512,
                              operand="activation")
    output = lower_linear_spad_mvout(profile, buffer="dst",
                                     total_bytes=16384, tile_cols=16,
                                     spad_row=0)
    assert [(len(p.commands), p.transferred_bytes, p.row_requests,
             p.minimum_buffer_bytes) for p in (a, b, scales, output)] == [
        (17, 16384, 256, 16384), (65, 16384, 1024, 16384),
        (1, 512, 64, 512), (65, 16384, 1024, 16384)]
    assert [(c.rs1.byte_offset, c.rs2.immediate & 0xffffffff)
            for c in a.commands[1:]] == [
        (i * 16 * 128 + k * 16, (i * 8 + k) * 16)
        for i in range(8) for k in (0, 4)]
    assert [(c.rs1.byte_offset, c.rs2.immediate & 0xffffffff)
            for c in b.commands[1:]] == [
        (i * 16 * 128 + k * 16, 1024 + (i * 8 + k) * 16)
        for i in range(8) for k in range(8)]
    assert [c.rs1.byte_offset for c in output.commands[1:]] == list(
        range(0, 16384, 256))
    assert [c.rs2.immediate & 0xffffffff for c in output.commands[1:]] == list(
        range(0, 1024, 16))
    assert scales.commands[0].funct == 27
    assert scales.commands[0].rs2.immediate == 512


def test_memory_phases_reject_unsupported_profile_or_placement() -> None:
    profile = load_profile(PROFILE)
    small = deepcopy(profile)
    small["resources"]["dma_max_bytes"] = 16
    with pytest.raises(ValueError, match="unsupported"):
        lower_matrix_mvin(small, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=0)
    with pytest.raises(ValueError, match="unsupported"):
        lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=16000)
    with pytest.raises(ValueError, match="exceeds"):
        lower_linear_spad_mvout(profile, buffer="dst", total_bytes=16384,
                                tile_cols=16, spad_row=16000)
    with pytest.raises(ValueError, match="cannot hold"):
        lower_scale_load(profile, buffer="src", payload_bytes=16384,
                         operand="activation")
