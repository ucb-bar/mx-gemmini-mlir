"""Exact FP6 codebook indexing for the selected 16x16 MX Gemmini RTL.

This transform accepts caller-selected E3M2 codebooks. It does not choose
codebooks, alter FP6 values, or assert hardware arithmetic equivalence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .layout import pack_fp6_lut
from .operands import checked_code_matrix, pack_nibble_operands


@dataclass(frozen=True)
class FP6IndexedContraction:
    m: int
    k: int
    n: int
    granularity_shift: int
    activation_bytes: bytes  # [M/2][K], low nibble = even M row
    weight_bytes: bytes  # [K][N/2], low nibble = even N column
    activation_lut_bytes: bytes
    weight_lut_bytes: bytes
    activation_lut_lines: int
    weight_lut_lines: int


def _checked_lines(
    lines: Sequence[Sequence[int]], *, axis: int, shift: int, operand: str
) -> tuple[dict[int, int], ...]:
    required = (axis + (1 << shift) - 1) >> shift
    if required > 64 or len(lines) != required:
        raise ValueError(f"{operand} needs exactly {required} LUT lines (at most 64)")
    lookup = []
    for line in lines:
        if len(line) != 16 or any(type(code) is not int or not 0 <= code < 64 for code in line):
            raise ValueError(f"{operand} LUT lines need sixteen 6-bit E3M2 codes")
        # Duplicate entries are legal; the lowest index gives stable encoding.
        lookup.append({code: next(i for i, value in enumerate(line) if value == code) for code in line})
    return tuple(lookup)


def pack_fp6_indexed_contraction(
    activation_codes: Sequence[Sequence[int]],
    weight_codes: Sequence[Sequence[int]],
    *,
    activation_lut: Sequence[Sequence[int]],
    weight_lut: Sequence[Sequence[int]],
    granularity_shift: int = 1,
) -> FP6IndexedContraction:
    """Map exact codes to nibble indices, then pack selected DIM16 operand layout.

    ``granularity_shift`` is CONFIG_SCALE_MEM rs2[15:0]: line for row/column
    ``axis_index >> granularity_shift``. The RTL's LUT read counters are 6 bits,
    so a single configured spatial window is restricted to M,N <= 128.
    Missing codes fail instead of silently changing numerical results.
    """
    activation_codes = checked_code_matrix(activation_codes, bits=6, name="activation")
    weight_codes = checked_code_matrix(weight_codes, bits=6, name="weight")
    activation_lut = checked_code_matrix(activation_lut, bits=6, name="activation LUT")
    weight_lut = checked_code_matrix(weight_lut, bits=6, name="weight LUT")
    m, k = len(activation_codes), len(activation_codes[0])
    weight_k, n = len(weight_codes), len(weight_codes[0])
    if k != weight_k:
        raise ValueError("activation K and weight K must agree")
    if m % 32 or n % 32 or k % 32 or m > 128 or n > 128:
        raise ValueError("selected FP6 layout needs M,N multiples of 32 within 128 and K multiple of 32")
    if type(granularity_shift) is not int or not 0 <= granularity_shift <= 6:
        raise ValueError("FP6 LUT granularity shift must be an integer from 0 to 6")
    a_lookup = _checked_lines(activation_lut, axis=m, shift=granularity_shift, operand="activation")
    b_lookup = _checked_lines(weight_lut, axis=n, shift=granularity_shift, operand="weight")

    a_indices: list[list[int]] = []
    for row, values in enumerate(activation_codes):
        lut = a_lookup[row >> granularity_shift]
        try:
            a_indices.append([lut[code] for code in values])
        except KeyError as exc:
            raise ValueError(f"activation code {exc.args[0]} missing from LUT line {row >> granularity_shift}") from exc
    b_indices: list[list[int]] = []
    for values in weight_codes:
        row_indices = []
        for col, code in enumerate(values):
            try:
                row_indices.append(b_lookup[col >> granularity_shift][code])
            except KeyError as exc:
                raise ValueError(f"weight code {exc.args[0]} missing from LUT line {col >> granularity_shift}") from exc
        b_indices.append(row_indices)

    activation_bytes, weight_bytes = pack_nibble_operands(a_indices, b_indices)
    return FP6IndexedContraction(
        m=m, k=k, n=n, granularity_shift=granularity_shift,
        activation_bytes=activation_bytes, weight_bytes=weight_bytes,
        activation_lut_bytes=pack_fp6_lut(activation_lut),
        weight_lut_bytes=pack_fp6_lut(weight_lut),
        activation_lut_lines=len(activation_lut),
        weight_lut_lines=len(weight_lut),
    )
