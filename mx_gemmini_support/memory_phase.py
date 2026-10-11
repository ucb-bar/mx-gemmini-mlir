"""Profile-checked physical DMA phases for standalone MX memory programs.

These commands describe transfers without inventing a contraction. A frontend
may use them to lower explicit transfer operations or benchmark phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .command_ir import Command, Operand
from .layout import scale_load_rs2
from .physical_program import _cmd, _config_ld, _config_st


@dataclass(frozen=True)
class MemoryPhase:
    commands: tuple[Command, ...]
    transferred_bytes: int
    row_requests: int
    minimum_buffer_bytes: int


def _capacity(profile: Mapping) -> tuple[int, int]:
    if (profile.get("schema") != "mx_gemmini.target_profile.v2" or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        raise ValueError("standalone MX DMA needs a selected DIM16 RoCC profile")
    resources = profile.get("resources", {})
    scratchpad = resources.get("scratchpad_bytes")
    maximum = resources.get("dma_max_bytes")
    if (type(scratchpad) is not int or scratchpad <= 0 or scratchpad % 16 or
            type(maximum) is not int or maximum < 16 or maximum % 16):
        raise ValueError("selected MX profile has no checked DMA capacity")
    return scratchpad // 16, maximum


def lower_matrix_mvin(profile: Mapping, *, buffer: str, matrix_rows: int,
                      matrix_cols: int, burst_cols: int,
                      spad_row: int) -> MemoryPhase:
    """Load a row-major byte matrix in DIM16-high, 16..DMA-max-wide tiles."""
    spad_limit, maximum = _capacity(profile)
    if (any(type(n) is not int for n in
            (matrix_rows, matrix_cols, burst_cols, spad_row)) or
            matrix_rows <= 0 or matrix_rows % 16 or
            matrix_cols <= 0 or matrix_cols % 16 or
            burst_cols < 16 or burst_cols % 16 or
            burst_cols > maximum or matrix_cols % burst_cols or
            spad_row < 0 or spad_row % 16 or
            spad_row + matrix_rows * matrix_cols // 16 > spad_limit):
        raise ValueError("MX matrix DMA shape or scratchpad placement is unsupported")
    # The source instruction accepts 16-bit cols/rows but a single DMA row
    # cannot exceed the profile's actual maximum transfer width.
    if burst_cols >= 1 << 16 or matrix_cols >= 1 << 32:
        raise ValueError("MX matrix DMA width exceeds the command field")
    tiles_per_row = matrix_cols // 16
    commands = [_config_ld(matrix_cols)]
    for i in range(matrix_rows // 16):
        for k in range(0, tiles_per_row, burst_cols // 16):
            offset = i * 16 * matrix_cols + k * 16
            row = spad_row + (i * tiles_per_row + k) * 16
            commands.append(_cmd(
                2, Operand(buffer=buffer, byte_offset=offset),
                (16 << 48) | (burst_cols << 32) | row))
    requests = matrix_rows * matrix_cols // burst_cols
    return MemoryPhase(tuple(commands), matrix_rows * matrix_cols,
                       requests, matrix_rows * matrix_cols)


def lower_linear_spad_mvout(profile: Mapping, *, buffer: str, total_bytes: int,
                            tile_cols: int, spad_row: int) -> MemoryPhase:
    """Store consecutive DIM16 scratchpad tiles into a flat byte buffer."""
    spad_limit, maximum = _capacity(profile)
    if any(type(n) is not int for n in (total_bytes, tile_cols, spad_row)):
        raise ValueError("MX linear scratchpad store needs integer geometry")
    tile_bytes = 16 * tile_cols
    if (
            tile_cols < 16 or tile_cols % 16 or tile_cols > maximum or
            total_bytes <= 0 or total_bytes % tile_bytes or
            spad_row < 0 or spad_row % 16 or
            spad_row + total_bytes // 16 > spad_limit):
        raise ValueError("MX linear scratchpad store exceeds the selected profile")
    commands = [_config_st(tile_cols)]
    for offset in range(0, total_bytes, tile_bytes):
        row = spad_row + offset // 16
        commands.append(_cmd(
            3, Operand(buffer=buffer, byte_offset=offset),
            (16 << 48) | (tile_cols << 32) | row))
    return MemoryPhase(tuple(commands), total_bytes,
                       total_bytes // tile_cols, total_bytes)


def lower_scale_load(profile: Mapping, *, buffer: str,
                     payload_bytes: int, operand: str) -> MemoryPhase:
    """Issue the source's one-dimensional MX scale-loader command."""
    _capacity(profile)
    config = profile.get("resources", {}).get("scale_mem_config")
    if (type(payload_bytes) is not int or
            not isinstance(config, dict) or
            type(config.get("size_bytes")) is not int or
            payload_bytes > config["size_bytes"] // 2):
        raise ValueError("selected MX scale memory cannot hold this operand")
    command = _cmd(27, Operand(buffer=buffer),
                   scale_load_rs2(payload_bytes, operand=operand))
    return MemoryPhase((command,), payload_bytes,
                       payload_bytes // 8, payload_bytes)
