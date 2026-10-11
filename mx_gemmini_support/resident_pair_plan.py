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


@dataclass(frozen=True)
class RectangularPairPlan:
    """MM1 M×K1 by K1×N1, followed by MM2 M×N1 by N1×N2."""

    m: int
    n: int  # N2, the final output width.
    k: int  # N1, the resident activation width and MM2 reduction.
    first_k: int
    dim: int
    rows: int
    a_row: int
    b1_row: int
    b_row: int  # B2, required by the resident MM2 command.
    c1_row: int
    c2_row: int
    a_rows: int
    b1_rows: int
    b_rows: int
    c1_rows: int
    c_rows: int
    m_tiles: int
    first_n_tiles: int
    first_k_tiles: int
    n_tiles: int
    k_tiles: int
    first_a_scale_bytes: int
    first_b_scale_bytes: int
    c1_scale_bytes: int
    b2_scale_bytes: int
    output_scale_bytes: int


def plan_fp8_rectangular_pair(profile: dict, *,
                              first_shape: tuple[int, int, int],
                              second_shape: tuple[int, int, int],
                              a_row: int, c1_row: int, c2_row: int
                              ) -> RectangularPairPlan:
    """Place both B matrices at the tail, reusing that region after MM1."""
    resources = profile["resources"]
    dim = profile["geometry"]["mesh_columns"]
    if (profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_rows"] != dim or dim != 16 or
            not resources.get("requantizer") or
            profile["name"] != "MxGemminiRocketConfig" or
            resources.get("spad_requant") or resources.get("vpu") or
            "fp8_e4m3" not in profile["candidate_output_modes"]):
        raise ValueError("rectangular pair needs Nicolas's plain DIM16 RoCC FP8 profile")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    if (len(first_shape) != 3 or len(second_shape) != 3 or
            any(type(value) is not int or value <= 0
                for value in (*first_shape, *second_shape))):
        raise ValueError("rectangular pair needs positive integer dimensions")
    m, n1, k1 = first_shape
    m2, n2, k2 = second_shape
    if (m != m2 or n1 != k2 or m % dim or
            any(value % 32 for value in (n1, n2, k1)) or
            any(value // dim > 0xffff for value in (m, n1, n2, k1))):
        raise ValueError("rectangular pair SSA shapes or scale groups differ")
    if (any(type(row) is not int or row < 0 or row % dim
            for row in (a_row, c1_row, c2_row)) or
            resources["scratchpad_bytes"] % dim or
            resources["scale_mem_config"]["size_bytes"] % 4):
        raise ValueError("rectangular pair target memory geometry is unsupported")
    rows = resources["scratchpad_bytes"] // dim
    if rows > 1 << 14:
        raise ValueError("rectangular pair exceeds the RoCC row address field")
    a_rows, b1_rows, b2_rows = m * k1 // dim, k1 * n1 // dim, n1 * n2 // dim
    c1_rows, c2_rows = m * n1 // dim, m * n2 // dim
    b1_row, b2_row = rows - b1_rows, rows - b2_rows
    ranges = {"A": (a_row, a_row + a_rows),
              "C1": (c1_row, c1_row + c1_rows),
              "C2": (c2_row, c2_row + c2_rows),
              "B tail": (min(b1_row, b2_row), rows)}
    if (min(b1_row, b2_row) < 0 or
            any(end > rows for _, end in ranges.values()) or
            any(not (left[1] <= right[0] or right[1] <= left[0])
                for i, left in enumerate(ranges.values())
                for right in list(ranges.values())[i + 1:])):
        raise ValueError("rectangular pair scratchpad row lifetimes overlap")
    scale_half = resources["scale_mem_config"]["size_bytes"] // 4
    a_scales, b1_scales, c1_scales = m * k1 // 32, k1 * n1 // 32, m * n1 // 32
    b2_scales, c2_scales = n1 * n2 // 32, m * n2 // 32
    if (max(a_scales, b1_scales, c1_scales, b2_scales, c2_scales) > scale_half or
            max(m * n1, m * n2) * 2 > resources["accumulator_bytes"]):
        raise ValueError("rectangular pair scale or accumulator capacity is exceeded")
    return RectangularPairPlan(
        m, n2, n1, k1, dim, rows, a_row, b1_row, b2_row, c1_row, c2_row,
        a_rows, b1_rows, b2_rows, c1_rows, c2_rows,
        m // dim, n1 // dim, k1 // dim, n2 // dim, n1 // dim,
        a_scales, b1_scales, c1_scales, b2_scales, c2_scales)


def lower_first_fp8_rectangular(plan: RectangularPairPlan, *,
                                activation_buffer: str,
                                activation_scales_buffer: str,
                                weight_buffer: str,
                                weight_scales_buffer: str,
                                output_scales_buffer: str
                                ) -> tuple[Command | Fence, ...]:
    """Use Nicolas's rectangular k-major B tile layout for MM1."""
    i, j, kk = plan.m_tiles, plan.first_n_tiles, plan.first_k_tiles
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0),
        _cmd(0, (1 << 16) | (1 << 2), 1 << 48),
        _cmd(27, Operand(buffer=activation_scales_buffer),
             plan.first_a_scale_bytes),
        _cmd(27, Operand(buffer=weight_scales_buffer),
             (1 << 32) | plan.first_b_scale_bytes),
        Fence(), _config_ld(plan.first_k),
    ]
    for mi in range(i):
        for ki in range(kk):
            commands.append(_transfer(
                2, activation_buffer,
                mi * plan.dim * plan.first_k + ki * plan.dim,
                plan.a_row + (mi * kk + ki) * plan.dim))
    commands.append(_config_ld(plan.k))
    for ki in range(kk):
        for nj in range(j):
            commands.append(_transfer(
                2, weight_buffer,
                ki * plan.dim * plan.k + nj * plan.dim,
                plan.b1_row + (ki * j + nj) * plan.dim))
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


def plan_fp8_resident_pair(profile: dict, *, shape: tuple[int, int, int],
                           a_row: int, c1_row: int, c2_row: int,
                           allow_a_c1_reuse: bool = False
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
    if allow_a_c1_reuse and (
            shape != (64, 64, 64) or
            (a_row, c1_row, c2_row) != (0, 128, 512) or
            profile["name"] != "MxGemminiRocketConfig"):
        raise ValueError("A/C1 row reuse is qualified only for Nicolas's 64³ chain")
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
    # MM1 has consumed A before it writes C1. Nicolas's 64³ source reuses
    # rows 128..255 for C1; all other pairs must remain disjoint.
    if (b_row < 0 or any(end > rows for _, end in ranges.values()) or
            any(not (left[1] <= right[0] or right[1] <= left[0])
                for i, (left_name, left) in enumerate(ranges.items())
                for right_name, right in list(ranges.items())[i + 1:]
                if not (allow_a_c1_reuse and
                        {left_name, right_name} == {"A", "C1"}))):
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


def _plan_packed_resident_pair(profile: dict, *, shape: tuple[int, int, int],
                               a_row: int, c1_row: int, c2_row: int,
                               precision: str) -> ResidentPairPlan:
    """Place Nicolas's packed direct or LUT-indexed square chain."""
    resources = profile["resources"]
    if (profile.get("transport") != "rocket_rocc" or
            profile["name"] != "MxGemminiRocketConfig" or
            profile["geometry"]["mesh_columns"] != 16 or
            profile["geometry"]["mesh_rows"] != 16 or
            not resources.get("requantizer") or
            resources.get("vpu") or resources.get("spad_requant")):
        raise ValueError("packed resident pair needs Nicolas's plain DIM16 MX profile")
    lut = precision == "fp6_e3m2"
    if lut and not resources.get("lut"):
        raise ValueError("FP6 resident pair needs target LUT memory")
    require_compute(profile, precision, precision, pe_mode=4 if lut else 0,
                    activation_projection="lut" if lut else "direct",
                    weight_projection="lut" if lut else "direct")
    if shape not in ((64, 64, 64), (128, 128, 128)):
        raise ValueError("packed resident pair needs Nicolas's 64- or 128-cubed source shape")
    size = shape[0]
    rows = resources["scratchpad_bytes"] // 16
    a_rows = b_rows = c_rows = size * size // 32
    b_row = rows - b_rows
    ranges = [(a_row, a_row + a_rows), (c1_row, c1_row + c_rows),
              (c2_row, c2_row + c_rows), (b_row, rows)]
    if (rows != 16384 or any(type(row) is not int or row < 0 or row % 16
                             for row in (a_row, c1_row, c2_row)) or
            any(end > rows for _, end in ranges) or
            any(left[1] > right[0] and right[1] > left[0]
                for i, left in enumerate(ranges)
                for right in ranges[i + 1:]) or
            resources["scale_mem_config"]["size_bytes"] // 4 < size * size // 32 or
            resources["accumulator_bytes"] < size * size * 2):
        raise ValueError("packed resident pair scratchpad or scale capacity differs")
    scales = size * size // 32
    return ResidentPairPlan(size, size, size, 16, rows, a_row, b_row, c1_row,
                            c2_row, a_rows, b_rows, c_rows,
                            size // 32, size // 32, size // 16,
                            scales, scales, scales)


def plan_fp4_resident_pair(profile: dict, *, shape: tuple[int, int, int],
                           a_row: int, c1_row: int, c2_row: int) -> ResidentPairPlan:
    return _plan_packed_resident_pair(
        profile, shape=shape, a_row=a_row, c1_row=c1_row, c2_row=c2_row,
        precision="fp4_e2m1")


def plan_fp6_resident_pair(profile: dict, *, shape: tuple[int, int, int],
                           a_row: int, c1_row: int, c2_row: int) -> ResidentPairPlan:
    return _plan_packed_resident_pair(
        profile, shape=shape, a_row=a_row, c1_row=c1_row, c2_row=c2_row,
        precision="fp6_e3m2")


def lower_first_fp4_resident(plan: ResidentPairPlan, *,
                             activation_buffer: str, activation_scales_buffer: str,
                             weight_buffer: str, weight_scales_buffer: str,
                             output_scales_buffer: str) -> tuple[Command | Fence, ...]:
    """Issue MM1 and leave its packed output and scales resident for MM2."""
    i, j, kk = plan.m_tiles, plan.n_tiles, plan.k_tiles
    config_ex = (1 << 16) | (2 << 14) | (2 << 12) | (2 << 10) | (1 << 2)
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0), _cmd(0, config_ex, 1 << 48),
        _cmd(27, Operand(buffer=activation_scales_buffer), plan.a_scale_bytes),
        _cmd(27, Operand(buffer=weight_scales_buffer),
             (1 << 32) | plan.b_scale_bytes),
        Fence(), _config_ld(plan.m),
    ]
    for mi in range(i):
        for ki in range(kk):
            commands.append(_transfer(
                2, activation_buffer, mi * plan.dim * plan.m + ki * plan.dim,
                plan.a_row + (mi * kk + ki) * plan.dim))
    commands.append(_config_ld(plan.n // 2))
    for ki in range(kk):
        for nj in range(j):
            commands.append(_transfer(
                2, weight_buffer, ki * plan.dim * plan.n // 2 + nj * plan.dim,
                plan.b_row + (ki * j + nj) * plan.dim))
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


def lower_first_fp6_resident(plan: ResidentPairPlan, *,
                             activation_buffer: str, activation_scales_buffer: str,
                             weight_buffer: str, weight_scales_buffer: str,
                             activation_lut_buffer: str, weight_lut_buffer: str,
                             output_lut_buffer: str,
                             output_scales_buffer: str) -> tuple[Command | Fence, ...]:
    """Issue LUT-indexed MM1 and leave packed C1 codes and scales resident."""
    i, j, kk = plan.m_tiles, plan.n_tiles, plan.k_tiles
    groups = plan.m // 2
    fmt = (1 << 16) | (1 << 14) | (1 << 12) | (1 << 10) | (1 << 2)
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0),
        _cmd(0, fmt | (1 << 4), 1 << 48),
        _cmd(0, fmt | (1 << 5), 1 << 48),
        _config_st(8),
        _cmd(29, Operand(buffer=weight_lut_buffer), (6 << 34) | groups),
        _cmd(29, Operand(buffer=activation_lut_buffer),
             (6 << 34) | (1 << 32) | groups),
        _cmd(29, Operand(buffer=output_lut_buffer),
             (6 << 34) | (2 << 32) | groups),
        _cmd(27, Operand(buffer=activation_scales_buffer), plan.a_scale_bytes),
        _cmd(27, Operand(buffer=weight_scales_buffer),
             (1 << 32) | plan.b_scale_bytes),
        Fence(),
        _cmd(26, Operand(buffer=output_scales_buffer,
                         address_mask=(1 << 33) - 1,
                         or_bits=(1 << 63) | (kk << 51) | (j << 42) | (i << 33)), 1),
        _config_ld(plan.k),
    ]
    for mi in range(i):
        for ki in range(kk):
            commands.extend((
                _transfer(2, activation_buffer,
                          mi * plan.dim * plan.k + ki * plan.dim,
                          plan.a_row + (mi * kk + ki) * plan.dim),
                Fence()))
    commands.append(_config_ld(plan.n // 2))
    for ki in range(kk):
        for nj in range(j):
            commands.extend((
                _transfer(2, weight_buffer,
                          ki * plan.dim * plan.n // 2 + nj * plan.dim,
                          plan.b_row + (ki * j + nj) * plan.dim),
                Fence()))
    commands.extend([
        _cmd(9, 0, (kk << 32) | (j << 16) | i),
        _cmd(24, plan.a_row, plan.rows),
        _cmd(8, 0, (plan.c1_row << 32) | 0x200 | 0x38 | (1 << 10)),
        Fence(),
    ])
    return tuple(commands)
