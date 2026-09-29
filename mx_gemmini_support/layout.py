"""MX Gemmini operand-side layout for the pinned standalone RTL.

Source: Gemmini f0167390b56fb315deea90ac1fc3983772e92d82, selected
GemminiMxFPConfigs.standaloneMxFPConfig, MxGen a27ce3cd81513210c21f971ec3977defd13fa21e.
This is a compiler prototype, not a simulator-validated implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .operands import checked_code_matrix


_FORMAT_CODE = {"mxfp8": 0, "mxfp6": 1, "mxfp4": 2}
_SCALE_WIDTH = {"mxfp8": 16, "mxfp6": 32, "mxfp4": 32}
_ACTIVE_WINDOW = 4096
_MAX_SCALE_LOOP_BOUND = (1 << 9) - 1


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


@dataclass(frozen=True)
class ScaleWave:
    """One K-block range whose scales fit both operand windows.

    This is a capacity/layout plan, not an executable loop schedule. In
    particular, accumulator lifetime and command ordering need RTL testing.
    """

    fmt: str
    tiles_i: int
    tiles_j: int
    total_blocks: int
    block_start: int
    block_stop: int  # exclusive

    @property
    def k_tile_start(self) -> int:
        return 2 * self.block_start

    @property
    def k_tiles(self) -> int:
        return 2 * (self.block_stop - self.block_start)

    @property
    def activation_rows(self) -> int:
        return (self.block_stop - self.block_start) * self.tiles_i

    @property
    def weight_rows(self) -> int:
        return (self.block_stop - self.block_start) * self.tiles_j


def plan_scale_waves(
    fmt: str, *, tiles_i: int, tiles_j: int, k_tiles: int,
    max_blocks_per_wave: int | None = None,
) -> tuple[ScaleWave, ...]:
    """Partition K at 32-element MX block boundaries under the 4 KiB windows.

    CONFIG_SCALE_MEM has 9-bit I/J/K bounds. Every wave starts its scale
    payload at local offset zero; the selected RTL has no scale read base.
    An optional smaller block cap forces a K split for compiler diagnostics.
    """
    width = _width(fmt)
    if any(type(n) is not int for n in (tiles_i, tiles_j, k_tiles)):
        raise ValueError("scale loop bounds must be integers")
    if not 1 <= tiles_i <= _MAX_SCALE_LOOP_BOUND or not 1 <= tiles_j <= _MAX_SCALE_LOOP_BOUND:
        raise ValueError("I and J scale loop bounds must fit nonzero 9-bit fields")
    if k_tiles <= 0 or k_tiles % 2:
        raise ValueError("K tiles must contain whole 32-element MX blocks")
    if max_blocks_per_wave is not None and (
        type(max_blocks_per_wave) is not int or max_blocks_per_wave <= 0
    ):
        raise ValueError("max_blocks_per_wave must be a positive integer")
    rows_per_window = _ACTIVE_WINDOW // width
    blocks_per_wave = min(
        rows_per_window // tiles_i,
        rows_per_window // tiles_j,
        _MAX_SCALE_LOOP_BOUND // 2,
    )
    if max_blocks_per_wave is not None:
        blocks_per_wave = min(blocks_per_wave, max_blocks_per_wave)
    if blocks_per_wave == 0:
        raise ValueError("one K block exceeds an operand's active scale window")
    total_blocks = k_tiles // 2
    return tuple(
        ScaleWave(fmt, tiles_i, tiles_j, total_blocks, start, min(start + blocks_per_wave, total_blocks))
        for start in range(0, total_blocks, blocks_per_wave)
    )


def pack_wave_scales(
    wave: ScaleWave,
    activation_rows: Sequence[Sequence[int]],
    weight_rows: Sequence[Sequence[int]],
) -> tuple[bytes, bytes]:
    """Slice global [K block][I/J tile] rows into zero-based wave payloads."""
    if len(activation_rows) != wave.total_blocks * wave.tiles_i:
        raise ValueError("activation rows do not match the full scale plan")
    if len(weight_rows) != wave.total_blocks * wave.tiles_j:
        raise ValueError("weight rows do not match the full scale plan")
    act = pack_scale_rows(
        wave.fmt, activation_rows[wave.block_start * wave.tiles_i : wave.block_stop * wave.tiles_i]
    )
    wgt = pack_scale_rows(
        wave.fmt, weight_rows[wave.block_start * wave.tiles_j : wave.block_stop * wave.tiles_j]
    )
    return act, wgt


@dataclass(frozen=True)
class ScaleWaveTransfer:
    wave: ScaleWave
    activation_bytes: bytes
    weight_bytes: bytes


def plan_scale_transfers(
    fmt: str,
    activation_scales: Sequence[Sequence[int]],
    weight_scales: Sequence[Sequence[int]],
    *,
    weight_layout: str = "ng",
    max_blocks_per_wave: int | None = None,
) -> tuple[ScaleWaveTransfer, ...]:
    """Lay out E8M0 scales from [M][G] and [N][G] or [G][N] matrices.

    G is K/32. Each wave's payload is ordered [K block][I/J tile][lane]
    and starts at byte offset zero. ``ng`` is TorchAO Linear's [N][G]
    weight buffer; ``gn`` is a functional B contraction's [G][N] buffer.
    This plans bytes, not executable DMA. ``max_blocks_per_wave`` may force
    smaller windows without exceeding the physical capacity bound.
    """
    width = _width(fmt)
    if weight_layout not in ("ng", "gn"):
        raise ValueError("weight scale layout must be ng or gn")
    activation_scales = checked_code_matrix(activation_scales, bits=8, name="activation scales")
    weight_scales = checked_code_matrix(weight_scales, bits=8, name="weight scales")
    m = len(activation_scales)
    blocks = len(activation_scales[0]) if m else 0
    n = len(weight_scales) if weight_layout == "ng" else (len(weight_scales[0]) if weight_scales else 0)
    if not m or not n or not blocks or m % width or n % width:
        raise ValueError(f"{fmt} scales need nonempty M/N multiples of {width} and K/32 groups")
    if any(len(row) != blocks for row in activation_scales):
        raise ValueError("activation and weight scales must share one K/32 group count")
    if weight_layout == "ng" and any(len(row) != blocks for row in weight_scales):
        raise ValueError("activation and weight scales must share one K/32 group count")
    if weight_layout == "gn" and (len(weight_scales) != blocks or any(len(row) != n for row in weight_scales)):
        raise ValueError("activation and weight scales must share one K/32 group count")
    tiles_i, tiles_j = m // width, n // width
    waves = plan_scale_waves(
        fmt, tiles_i=tiles_i, tiles_j=tiles_j, k_tiles=2 * blocks,
        max_blocks_per_wave=max_blocks_per_wave,
    )
    a_rows = [
        [activation_scales[tile * width + lane][block] for lane in range(width)]
        for block in range(blocks) for tile in range(tiles_i)
    ]
    b_rows = [
        [weight_scales[tile * width + lane][block] if weight_layout == "ng"
         else weight_scales[block][tile * width + lane] for lane in range(width)]
        for block in range(blocks) for tile in range(tiles_j)
    ]
    return tuple(
        ScaleWaveTransfer(wave, *pack_wave_scales(wave, a_rows, b_rows))
        for wave in waves
    )


def config_scale_mem_rs1(
    *, tiles_i: int, tiles_j: int, k_tiles: int,
    activation_buffer: int = 0, weight_buffer: int = 0,
    reset_requantizer: bool = False, resident: bool = False,
    output_scale_address: int = 0,
) -> int:
    """Encode selected CONFIG_SCALE_MEM rs1, including residency at bit 63.

    Bit 62 is wired to the requantizer counter reset; it does not directly
    reset ScaleFactorMem's read counters in the pinned RTL.
    """
    if any(type(n) is not int or not 1 <= n <= _MAX_SCALE_LOOP_BOUND for n in (tiles_i, tiles_j, k_tiles)):
        raise ValueError("I/J/K bounds must fit nonzero 9-bit fields")
    if activation_buffer not in (0, 1) or weight_buffer not in (0, 1):
        raise ValueError("scale buffer selectors must be 0 or 1")
    if type(output_scale_address) is not int or not 0 <= output_scale_address < (1 << 33):
        raise ValueError("output scale address must fit 33 bits")
    return (
        output_scale_address
        | (tiles_i << 33)
        | (tiles_j << 42)
        | (k_tiles << 51)
        | (activation_buffer << 60)
        | (weight_buffer << 61)
        | (int(reset_requantizer) << 62)
        | (int(resident) << 63)
    )


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
