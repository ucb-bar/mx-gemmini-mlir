"""Selected DIM16 MX operand bytes, separate from numerical quantization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def matrix_shape(codes: Sequence[Sequence[int]], *, bits: int, name: str) -> tuple[int, int]:
    rows = len(codes)
    cols = len(codes[0]) if rows else 0
    if rows == 0 or cols == 0 or any(len(row) != cols for row in codes):
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    if any(type(code) is not int or not 0 <= code < (1 << bits) for row in codes for code in row):
        raise ValueError(f"{name} must contain {bits}-bit element codes")
    return rows, cols


def pack_nibble_operands(
    activation: Sequence[Sequence[int]], weight: Sequence[Sequence[int]]
) -> tuple[bytes, bytes]:
    """A [M/2][K] pairs M rows; B [K][N/2] pairs N columns.

    The even row/column is the low nibble. Shapes and values must already be
    checked by the caller. Used for direct FP4 and FP6 LUT indices.
    """
    m, k, n = len(activation), len(activation[0]), len(weight[0])
    a = bytes(
        activation[2 * pair][inner] | (activation[2 * pair + 1][inner] << 4)
        for pair in range(m // 2) for inner in range(k)
    )
    b = bytes(
        weight[inner][2 * pair] | (weight[inner][2 * pair + 1] << 4)
        for inner in range(k) for pair in range(n // 2)
    )
    return a, b


@dataclass(frozen=True)
class DirectOperands:
    fmt: str
    m: int
    k: int
    n: int
    activation_bytes: bytes
    weight_bytes: bytes


def pack_direct_operands(
    fmt: str, activation_codes: Sequence[Sequence[int]], weight_codes: Sequence[Sequence[int]]
) -> DirectOperands:
    """Encode FP8 direct bytes or FP4 direct nibbles for one logical MxKxN.

    This only determines source-buffer layout. Scale-window capacity,
    scratchpad allocation, DMA commands and executable loops are separate.
    """
    if fmt not in ("mxfp8", "mxfp4"):
        raise ValueError("direct operand packing supports selected MXFP8 and MXFP4")
    bits = 8 if fmt == "mxfp8" else 4
    m, k = matrix_shape(activation_codes, bits=bits, name="activation")
    weight_k, n = matrix_shape(weight_codes, bits=bits, name="weight")
    if k != weight_k:
        raise ValueError("activation K and weight K must agree")
    spatial_tile = 16 if fmt == "mxfp8" else 32
    if m % spatial_tile or n % spatial_tile or k % 32:
        raise ValueError(f"{fmt} needs M/N multiples of {spatial_tile} and K multiple of 32")
    if fmt == "mxfp8":
        a = bytes(code for row in activation_codes for code in row)
        b = bytes(code for row in weight_codes for code in row)
    else:
        a, b = pack_nibble_operands(activation_codes, weight_codes)
    return DirectOperands(fmt, m, k, n, a, b)
