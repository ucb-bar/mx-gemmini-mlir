"""Independent BF16-to-MXFP8 output reference for Nicolas's current convention."""

from __future__ import annotations

from bisect import bisect_left
import math
import struct


_POSITIVE_E4M3 = tuple(
    (code, (mantissa * 2.0 ** -9 if exponent == 0
            else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)))
    for code in range(0x7f)
    for exponent, mantissa in [(code >> 3, code & 7)]
)
_VALUES = tuple(value for _, value in _POSITIVE_E4M3)
_E3M1_VALUES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75,
                1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
_E3M1_TO_E2M1 = (0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7, 7)


def _e4m3_rne(value: float) -> int:
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError("FP8 output reference requires finite BF16 values")
    if magnitude >= _VALUES[-1]:
        return sign | 0x7e
    upper = bisect_left(_VALUES, magnitude)
    if upper == 0:
        return sign
    lower = upper - 1
    low_gap, high_gap = magnitude - _VALUES[lower], _VALUES[upper] - magnitude
    code = (lower if low_gap < high_gap or (low_gap == high_gap and lower % 2 == 0)
            else upper)
    return sign | code


def _e2m1_via_e3m1_rne(value: float) -> int:
    """Project a scaled BF16 through the E3M1 intermediate used by FP4."""
    sign = 0x8 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError("FP4 output reference requires finite BF16 values")
    if magnitude >= _E3M1_VALUES[-1]:
        return sign | 7
    upper = bisect_left(_E3M1_VALUES, magnitude)
    if upper == 0:
        return 0
    lower = upper - 1
    low_gap, high_gap = magnitude - _E3M1_VALUES[lower], _E3M1_VALUES[upper] - magnitude
    index = (lower if low_gap < high_gap or (low_gap == high_gap and lower % 2 == 0)
             else upper)
    code = _E3M1_TO_E2M1[index]
    return sign | code if code else 0


def quantize_bf16_fp8_output(bf16_bytes: bytes, m: int, n: int) -> tuple[bytes, bytes]:
    """Use MXQuant's po2 scale, hardware 2^-23 floor, and RNE E4M3 grid.

    This deliberately starts from the source BF16 golden and does not call
    Spike or the RTL model. E8M0 scales are row-major [M][N/32].
    """
    if m <= 0 or n <= 0 or n % 32 or len(bf16_bytes) != 2 * m * n:
        raise ValueError("BF16 output reference shape or byte count is invalid")
    values = [struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
              for (word,) in struct.iter_unpack("<H", bf16_bytes)]
    codes = bytearray(m * n)
    scales = bytearray(m * n // 32)
    for row in range(m):
        for group in range(n // 32):
            begin = row * n + group * 32
            block = values[begin:begin + 32]
            if any(not math.isfinite(value) for value in block):
                raise ValueError("FP8 output reference requires finite BF16 values")
            maximum = max(abs(value) for value in block)
            exponent = math.frexp(max(maximum, 2.0 ** -23))[1] - 1
            scale_code = max(0, min(254, exponent + 127))
            scale = math.ldexp(1.0, scale_code - 127)
            scales[row * (n // 32) + group] = scale_code
            for offset, value in enumerate(block):
                codes[begin + offset] = _e4m3_rne(value / scale)
    return bytes(codes), bytes(scales)


def quantize_bf16_fp4_output(bf16_bytes: bytes, m: int, n: int) -> tuple[bytes, bytes]:
    """Use Nicolas's E2M1 output projection and packed-even/odd-M layout."""
    if m <= 0 or m % 2 or n <= 0 or n % 32 or len(bf16_bytes) != 2 * m * n:
        raise ValueError("BF16 FP4 output reference shape or byte count is invalid")
    values = [struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
              for (word,) in struct.iter_unpack("<H", bf16_bytes)]
    packed = bytearray(m * n // 2)
    scales = bytearray(m * n // 32)
    for row in range(m):
        for group in range(n // 32):
            begin = row * n + group * 32
            block = values[begin:begin + 32]
            if any(not math.isfinite(value) for value in block):
                raise ValueError("FP4 output reference requires finite BF16 values")
            maximum = max(abs(value) for value in block)
            scale_code = (0 if maximum == 0 else
                          max(0, min(254, math.frexp(maximum)[1] - 1 + 127)))
            scale = math.ldexp(1.0, scale_code - 127)
            scales[row * (n // 32) + group] = scale_code
            for col, value in enumerate(block, group * 32):
                code = _e2m1_via_e3m1_rne(value / scale)
                position = (row // 2) * n + col
                packed[position] |= code << (4 if row & 1 else 0)
    return bytes(packed), bytes(scales)
