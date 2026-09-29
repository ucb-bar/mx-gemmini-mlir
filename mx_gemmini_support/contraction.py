"""Coherent MX contraction payloads for the pinned standalone RTL.

This composes source-buffer representations only. No RoCC command sequence,
scratchpad allocation, or accelerator numerical result is certified here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .fp6 import pack_fp6_indexed_contraction
from .layout import ScaleWave, plan_scale_transfers
from .operands import checked_code_matrix, pack_direct_operands


@dataclass(frozen=True)
class MxWavePayload:
    wave: ScaleWave
    activation_bytes: bytes
    weight_bytes: bytes
    activation_scale_bytes: bytes
    weight_scale_bytes: bytes


@dataclass(frozen=True)
class MxContractionPayload:
    fmt: str
    m: int
    k: int
    n: int
    waves: tuple[MxWavePayload, ...]
    activation_lut_bytes: bytes = b""
    weight_lut_bytes: bytes = b""
    lut_lines_per_operand: tuple[int, int] = (0, 0)
    lut_granularity_shift: int | None = None


def plan_mx_contraction_payload(
    fmt: str,
    activation_codes: Sequence[Sequence[int]],
    weight_codes: Sequence[Sequence[int]],
    activation_scales: Sequence[Sequence[int]],
    weight_scales: Sequence[Sequence[int]],
    *,
    weight_scale_layout: str = "ng",
    activation_lut: Sequence[Sequence[int]] | None = None,
    weight_lut: Sequence[Sequence[int]] | None = None,
    lut_granularity_shift: int = 1,
    max_blocks_per_wave: int | None = None,
) -> MxContractionPayload:
    """Slice matching operand and E8M0 bytes into capacity-bounded K waves.

    A is [M][K], B is [K][N], and activation scales are [M][K/32].
    Weight scales may be [N][K/32] (``ng``) or [K/32][N] (``gn``).
    FP6 requires caller-selected exact E3M2 codebooks. A smaller explicit
    ``max_blocks_per_wave`` can force a compiler diagnostic K split.
    """
    bits = {"mxfp8": 8, "mxfp6": 6, "mxfp4": 4}.get(fmt)
    if bits is None:
        raise ValueError(f"unsupported selected MX format: {fmt}")
    a_codes = checked_code_matrix(activation_codes, bits=bits, name="activation")
    b_codes = checked_code_matrix(weight_codes, bits=bits, name="weight")
    m, k = len(a_codes), len(a_codes[0])
    weight_k, n = len(b_codes), len(b_codes[0])
    if k != weight_k or k % 32:
        raise ValueError("matching A/B contraction K must contain whole 32-element blocks")
    if fmt == "mxfp6" and (activation_lut is None or weight_lut is None):
        raise ValueError("MXFP6 requires activation and weight codebooks")
    if fmt != "mxfp6" and (activation_lut is not None or weight_lut is not None):
        raise ValueError("direct MX formats do not use operand codebooks")

    scale_transfers = plan_scale_transfers(
        fmt, activation_scales, weight_scales, weight_layout=weight_scale_layout,
        max_blocks_per_wave=max_blocks_per_wave,
    )
    width = 16 if fmt == "mxfp8" else 32
    first = scale_transfers[0].wave
    if first.tiles_i * width != m or first.tiles_j * width != n or first.total_blocks * 32 != k:
        raise ValueError("operand and E8M0 scale shapes disagree")

    waves: list[MxWavePayload] = []
    a_lut_bytes = b_lut_bytes = b""
    lut_lines = (0, 0)
    for transfer in scale_transfers:
        wave = transfer.wave
        k0, k1 = wave.block_start * 32, wave.block_stop * 32
        a_slice = tuple(row[k0:k1] for row in a_codes)
        b_slice = b_codes[k0:k1]
        if fmt == "mxfp6":
            packed = pack_fp6_indexed_contraction(
                a_slice, b_slice,
                activation_lut=activation_lut,
                weight_lut=weight_lut,
                granularity_shift=lut_granularity_shift,
            )
            if not waves:
                a_lut_bytes, b_lut_bytes = packed.activation_lut_bytes, packed.weight_lut_bytes
                lut_lines = (packed.activation_lut_lines, packed.weight_lut_lines)
        else:
            packed = pack_direct_operands(fmt, a_slice, b_slice)
        waves.append(MxWavePayload(
            wave,
            packed.activation_bytes,
            packed.weight_bytes,
            transfer.activation_bytes,
            transfer.weight_bytes,
        ))
    return MxContractionPayload(
        fmt, m, k, n, tuple(waves), a_lut_bytes, b_lut_bytes, lut_lines,
        lut_granularity_shift if fmt == "mxfp6" else None,
    )
