"""MX Gemmini operand-side layout for the pinned standalone RTL.

Source: Gemmini f0167390b56fb315deea90ac1fc3983772e92d82, selected
GemminiMxFPConfigs.standaloneMxFPConfig, MxGen a27ce3cd81513210c21f971ec3977defd13fa21e.
This is a compiler prototype, not a simulator-validated implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


_FORMAT_CODE = {"mxfp8": 0, "mxfp6": 1, "mxfp4": 2}
_SCALE_WIDTH = {"mxfp8": 16, "mxfp6": 32, "mxfp4": 32}
_ACTIVE_WINDOW = 4096


def _width(fmt: str) -> int:
    try:
        return _SCALE_WIDTH[fmt]
    except KeyError as exc:
        raise ValueError(f"unsupported selected MX format: {fmt}") from exc


def config_format_code(fmt: str) -> int:
    """CONFIG_EX activation/weight code; output code 3 disables requantization."""
    _width(fmt)
    return _FORMAT_CODE[fmt]


def scale_row_index(*, tile: int, k_tile: int, tiles_axis: int) -> int:
    """ScaleFactorMem read address before bank selection for DIM16.

    Its K counter advances per 16-element tile, while a block covers 32.
    tiles_axis is CONFIG_SCALE_MEM tiles_I for activation or tiles_J for weight.
    """
    if tiles_axis <= 0 or tiles_axis > 511 or tile < 0 or tile >= tiles_axis or k_tile < 0:
        raise ValueError("scale tile indices exceed selected DIM16 loop bounds")
    return (k_tile >> 1) * tiles_axis + tile


@dataclass(frozen=True)
class ScaleLocation:
    bank: int  # 0..3 within the selected activation or weight half
    row: int  # 0..127 within a 16-byte bank
    lane: int  # 0..15 within a bank row
    upload_offset: int  # byte offset from the operand's funct-27 DMA source


def scale_physical_location(fmt: str, logical_row: int, lane: int, *, buffer: int = 0) -> ScaleLocation:
    """Resolve one E8M0 byte through ScaleFactorMem's write address bits."""
    width = _width(fmt)
    limit = _ACTIVE_WINDOW // width
    if buffer not in (0, 1) or logical_row < 0 or logical_row >= limit:
        raise ValueError(f"{fmt} scale row exceeds one active buffer ({limit} rows)")
    if lane < 0 or lane >= width:
        raise ValueError(f"{fmt} scale lane must be below {width}")
    offset = buffer * _ACTIVE_WINDOW + logical_row * width + lane
    if fmt == "mxfp8":
        bank = (offset >> 11) & 3
        row = (offset >> 4) & 127
    else:
        bank = 2 * ((offset >> 12) & 1) + ((offset >> 4) & 1)
        row = (offset >> 5) & 127
    return ScaleLocation(bank, row, lane & 15, offset)


def pack_scale_rows(fmt: str, rows: Sequence[Sequence[int]]) -> bytes:
    """Pack buffer zero in logical K-block-major row order.

    MX_LOAD_SCALES always writes from local byte offset zero. This routine
    refuses buffer one; a future streaming planner must preserve or refill
    buffer zero before addressing the second 4 KiB window.
    """
    width = _width(fmt)
    if not rows or len(rows) > _ACTIVE_WINDOW // width:
        raise ValueError(f"{fmt} scale rows exceed the active 4 KiB window")
    payload = bytearray()
    for row in rows:
        if len(row) != width or any(type(code) is not int or not 0 <= code <= 255 for code in row):
            raise ValueError(f"each {fmt} scale row needs {width} E8M0 bytes")
        payload.extend(row)
    return bytes(payload)


def scale_load_rs2(payload_bytes: int, *, operand: str) -> int:
    """MX_LOAD_SCALES rs2: bit 32 selects weight; low 32 bits are byte count."""
    if payload_bytes <= 0 or payload_bytes > _ACTIVE_WINDOW or payload_bytes % 8:
        raise ValueError("scale payload must fit one active window and be an 8-byte multiple")
    if operand not in ("activation", "weight"):
        raise ValueError("scale operand must be activation or weight")
    return (int(operand == "weight") << 32) | payload_bytes


def pack_fp6_lut(lines: Sequence[Sequence[int]]) -> bytes:
    """Pack 16 E3M2 codes per line, low-bit first, padded for 8-byte DMA Gets."""
    if not lines or len(lines) > 64:
        raise ValueError("selected FP6 LUT holds 1..64 lines per operand")
    value = 0
    bit = 0
    for line in lines:
        if len(line) != 16 or any(type(code) is not int or not 0 <= code < 64 for code in line):
            raise ValueError("each FP6 LUT line needs sixteen 6-bit codes")
        for code in line:
            value |= code << bit
            bit += 6
    payload = value.to_bytes((bit + 7) // 8, "little")
    return payload + bytes((-len(payload)) % 8)


def lut_load_rs2(num_lines: int, *, operand: str) -> int:
    """MX_LOAD_LUT rs2: bits 39:34 entry width, 33:32 selector, 31:0 count."""
    if not 1 <= num_lines <= 64:
        raise ValueError("selected FP6 LUT holds 1..64 lines per operand")
    if operand not in ("activation", "weight"):
        raise ValueError("LUT operand must be activation or weight")
    selector = 1 if operand == "activation" else 0
    return (6 << 34) | (selector << 32) | num_lines
