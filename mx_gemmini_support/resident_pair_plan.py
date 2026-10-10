"""Plan two FP8 MX contractions with a live requantized scratchpad edge.

This checks target capacity and row lifetimes. Numerical qualification of a
particular shape and profile belongs to the caller's source/Spike receipt.
"""

from __future__ import annotations

from dataclasses import dataclass

from .command_ir import Command, Fence, Operand
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .target_profile import require_compute


@dataclass(frozen=True)
class ResidentPairPlan:
    m: int
    n: int
    k: int
    dim: int
    rows: int
    a_row: int
    b_row: int
    c1_row: int
    c2_row: int
    a_rows: int
    b_rows: int
    c_rows: int
    m_tiles: int
    n_tiles: int
    k_tiles: int
    a_scale_bytes: int
    b_scale_bytes: int
    output_scale_bytes: int


def plan_fp8_resident_pair(profile: dict, *, shape: tuple[int, int, int],
                           a_row: int, c1_row: int, c2_row: int
                           ) -> ResidentPairPlan:
    """Place full A/B/C1/C2 tiles; reuse B rows only after MM1 retires."""
    resources = profile["resources"]
    dim = profile["geometry"]["mesh_columns"]
    if (profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_rows"] != dim or dim != 16 or
            not resources.get("requantizer") or
            "fp8_e4m3" not in profile["candidate_output_modes"]):
        raise ValueError("resident pair needs a DIM16 RoCC FP8 requantizer profile")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    if (len(shape) != 3 or
            any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("resident pair needs positive integer M/N/K dimensions")
    m, n, k = shape
    if (m % dim or n % 32 or k % 32 or
            any(value // dim > 0xffff for value in shape)):
        raise ValueError("resident pair needs complete DIM16 and E8M0 blocks")
    if (any(type(row) is not int or row < 0 or row % dim for row in
            (a_row, c1_row, c2_row))):
        raise ValueError("resident pair scratchpad rows need DIM16 alignment")
    if (resources["scratchpad_bytes"] % dim or
            resources["scale_mem_config"]["size_bytes"] % 4):
        raise ValueError("resident pair target memory geometry is unsupported")
    rows = resources["scratchpad_bytes"] // dim
    if rows > 1 << 14:
        raise ValueError("resident pair scratchpad exceeds the RoCC row address field")
    a_rows, b_rows, c_rows = (m * k // dim, k * n // dim, m * n // dim)
    b_row = rows - b_rows
    ranges = {"A": (a_row, a_row + a_rows),
              "B": (b_row, rows),
              "C1": (c1_row, c1_row + c_rows),
              "C2": (c2_row, c2_row + c_rows)}
    if (b_row < 0 or any(end > rows for _, end in ranges.values()) or
            any(not (left[1] <= right[0] or right[1] <= left[0])
                for i, left in enumerate(ranges.values())
                for right in list(ranges.values())[i + 1:])):
        raise ValueError("resident pair scratchpad row lifetimes overlap")
    scale_half = resources["scale_mem_config"]["size_bytes"] // 4
    a_scales, b_scales, output_scales = (m * k // 32, k * n // 32, m * n // 32)
    if (max(a_scales, b_scales, output_scales) > scale_half or
            m * n * 2 > resources["accumulator_bytes"]):
        raise ValueError("resident pair scale or accumulator capacity is exceeded")
    return ResidentPairPlan(
        m, n, k, dim, rows, a_row, b_row, c1_row, c2_row,
        a_rows, b_rows, c_rows, m // dim, n // dim, k // dim,
        a_scales, b_scales, output_scales)


def lower_first_fp8_resident(plan: ResidentPairPlan, *,
                            activation_buffer: str, activation_scales_buffer: str,
                            weight_buffer: str, weight_scales_buffer: str,
                            output_scales_buffer: str) -> tuple[Command | Fence, ...]:
    """Emit MM1; leave tiled C1 and its activation scales live for MM2."""
    if plan.n != plan.k:
        raise ValueError("resident pair source weight layout needs square N/K")
    i, j, kk = plan.m_tiles, plan.n_tiles, plan.k_tiles
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0),
        _cmd(0, (1 << 16) | (1 << 2), 1 << 48),
        _cmd(27, Operand(buffer=activation_scales_buffer), plan.a_scale_bytes),
        _cmd(27, Operand(buffer=weight_scales_buffer),
             (1 << 32) | plan.b_scale_bytes),
        Fence(), _config_ld(plan.k),
    ]
    for mi in range(i):
        for ki in range(kk):
            commands.append(_transfer(
                2, activation_buffer, mi * plan.dim * plan.k + ki * plan.dim,
                plan.a_row + (mi * kk + ki) * plan.dim))
    for nj in range(j):
        for ki in range(kk):
            commands.append(_transfer(
                2, weight_buffer, nj * plan.dim * plan.n + ki * plan.dim,
                plan.b_row + (nj * kk + ki) * plan.dim))
    commands.extend([
        Fence(), _config_st(2),
        _cmd(26, Operand(buffer=output_scales_buffer,
                         address_mask=(1 << 33) - 1,
                         or_bits=(1 << 63) | (kk << 51) | (j << 42) | (i << 33)), 1),
        _cmd(9, 0, (kk << 32) | (j << 16) | i),
        _cmd(24, plan.a_row, plan.rows),
        _cmd(8, 0, (plan.c1_row << 32) | 0x200 | 0x38 | (1 << 10)),
        Fence(),
    ])
    return tuple(commands)
