"""Independent BF16-to-MX output references for Nicolas's current convention."""

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
_E3M2_VALUES = tuple(
    (mantissa * 0.0625 if exponent == 0 else
     (1.0 + mantissa * 0.25) * 2.0 ** (exponent - 3))
    for exponent in range(8) for mantissa in range(4))


def exact_bf16_x2(source: bytes) -> bytes:
    """Shift finite BF16 normals by one exponent, preserving signed zero."""
    if len(source) % 2:
        raise ValueError("BF16 x2 reference requires complete 16-bit values")
    output = bytearray()
    for offset in range(0, len(source), 2):
        word = int.from_bytes(source[offset:offset + 2], "little")
        exponent = (word >> 7) & 255
        if exponent == 0 and word & 0x7fff == 0:
            doubled = word
        elif 1 <= exponent <= 253:
            doubled = word + 0x80
        else:
            raise ValueError("exact BF16 x2 golden needs finite normal or zero source values")
        output.extend(doubled.to_bytes(2, "little"))
    return bytes(output)


def _bf16_scalar_arithmetic(source: bytes, scalar_bf16: int,
                            operation: str) -> bytes:
    """Apply Nicolas's double arithmetic followed by BF16 RNE."""
    if len(source) % 2:
        raise ValueError("BF16 scalar reference requires complete 16-bit values")
    if (not isinstance(scalar_bf16, int) or isinstance(scalar_bf16, bool) or
            not 0 <= scalar_bf16 <= 0xffff or
            (scalar_bf16 & 0x7f80) == 0x7f80):
        raise ValueError("BF16 scalar must be finite BF16 bits")
    if operation not in {"add", "mul"}:
        raise ValueError("BF16 scalar reference operation is unsupported")

    def as_float(word: int) -> float:
        return struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]

    scalar = as_float(scalar_bf16)
    output = bytearray()
    for (word,) in struct.iter_unpack("<H", source):
        value = as_float(word)
        if not math.isfinite(value):
            raise ValueError("BF16 scalar reference needs finite source values")
        result = value + scalar if operation == "add" else value * scalar
        bits = struct.unpack("<Q", struct.pack("<d", result))[0]
        sign = (bits >> 48) & 0x8000
        exponent = (bits >> 52) & 0x7ff
        if exponent == 0:
            rounded = sign
        else:
            e = exponent - 1023
            mantissa = (1 << 52) | (bits & ((1 << 52) - 1))
            shift = 45 if e >= -126 else -81 - e
            if shift >= 54:
                n = 0
            else:
                n = mantissa >> shift
                remainder = mantissa & ((1 << shift) - 1)
                half = 1 << (shift - 1)
                if remainder > half or (remainder == half and (n & 1)):
                    n += 1
            magnitude = (((e + 127) << 7) + n - 128 if e >= -126 else n)
            rounded = sign | min(magnitude, 0x7f80)
        output.extend(rounded.to_bytes(2, "little"))
    return bytes(output)


def bf16_mul_scalar(source: bytes, scalar_bf16: int) -> bytes:
    """Match Nicolas vpu_ref.h MULS, including final BF16 RNE."""
    return _bf16_scalar_arithmetic(source, scalar_bf16, "mul")


def bf16_add_scalar(source: bytes, scalar_bf16: int) -> bytes:
    """Match Nicolas vpu_ref.h ADDS, including final BF16 RNE."""
    return _bf16_scalar_arithmetic(source, scalar_bf16, "add")


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


def _radiance_e4m3_code(value: float) -> int:
    """The legacy Radiance mx_golden.cpp FP8 encoder, including its zero floor."""
    if value == 0.0 or not math.isfinite(value):
        return 0
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    exponent = math.floor(math.log2(magnitude))
    if exponent < -6:
        return 0
    if exponent > 8:
        exponent, mantissa = 8, 6
    else:
        mantissa = round((magnitude - math.ldexp(1.0, exponent)) /
                         math.ldexp(1.0, exponent - 3))
        if mantissa >= 8:
            exponent += 1
            mantissa = 0
            if exponent > 8:
                exponent, mantissa = 8, 6
        else:
            mantissa = min(mantissa, 6 if exponent == 8 else 7)
    return sign | (((exponent + 7) & 15) << 3) | mantissa


def quantize_bf16_radiance_header_fp8(bf16_bytes: bytes, m: int,
                                      n: int) -> tuple[bytes, bytes]:
    """Reproduce Radiance's FP8 C_out and row-major C_scales_row.

    Radiance's generator writes FP8 C_out for FP4 input kernels too. This
    deliberately models the source header, not the MX hardware requantizer.
    """
    if m <= 0 or n <= 0 or n % 32 or len(bf16_bytes) != 2 * m * n:
        raise ValueError("Radiance FP8 output shape or byte count is invalid")
    values = [struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
              for (word,) in struct.iter_unpack("<H", bf16_bytes)]
    codes = bytearray(m * n)
    scales = bytearray(m * n // 32)
    for row in range(m):
        for group in range(n // 32):
            begin = row * n + group * 32
            block = values[begin:begin + 32]
            if any(not math.isfinite(value) for value in block):
                raise ValueError("Radiance FP8 output requires finite BF16 values")
            maximum = max(abs(value) for value in block)
            scale_code = (0 if maximum == 0 else
                          max(0, min(254, math.floor(math.log2(maximum)) - 8 + 127)))
            scale = math.ldexp(1.0, scale_code - 127)
            scales[row * (n // 32) + group] = scale_code
            for offset, value in enumerate(block):
                codes[begin + offset] = _radiance_e4m3_code(value / scale)
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


def _bf16_rne(value: float) -> float:
    """Round an FP32 value exactly as the Spike FP6 postpass rounds to BF16."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
    return struct.unpack("<f", (rounded << 16).to_bytes(4, "little"))[0]


def _e3m2_rne(value: float) -> int:
    sign = 0x20 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError("FP6 output reference requires finite BF16 values")
    if magnitude >= _E3M2_VALUES[-1]:
        return sign | 31
    upper = bisect_left(_E3M2_VALUES, magnitude)
    if upper == 0:
        return sign
    lower = upper - 1
    low_gap, high_gap = magnitude - _E3M2_VALUES[lower], _E3M2_VALUES[upper] - magnitude
    chosen = lower if low_gap < high_gap or (low_gap == high_gap and lower % 2 == 0) else upper
    return sign | chosen


def _e3m2_fixed(code: int) -> int:
    exponent, mantissa = (code >> 2) & 7, code & 3
    if exponent == 0 and mantissa == 0:
        return 0
    signed_exponent = -2 if exponent == 0 else exponent - 3
    significand = (0 if exponent == 0 else 4) | mantissa
    magnitude = (significand << ((signed_exponent + 2) & 7)) & 0xff
    return -magnitude if code & 0x20 else magnitude


def quantize_bf16_fp6_lut_output(bf16_bytes: bytes, m: int, n: int,
                                  output_lut_bytes: bytes) -> tuple[bytes, bytes]:
    """Project each row pair onto its C LUT and pack indices along M.

    The finder measures distance in Nicolas's 9-bit fixed representation and
    resolves ties by the lowest LUT index. Output E8M0 scales are row-major.
    """
    if (m <= 0 or m % 2 or m > 128 or n <= 0 or n % 32 or
            len(bf16_bytes) != 2 * m * n or len(output_lut_bytes) != 64 * 12):
        raise ValueError("BF16 FP6 output reference shape or LUT byte count is invalid")
    lines = []
    for pair in range(64):
        packed = int.from_bytes(output_lut_bytes[pair * 12:(pair + 1) * 12], "little")
        lines.append(tuple((packed >> (6 * index)) & 0x3f for index in range(16)))
    values = [struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
              for (word,) in struct.iter_unpack("<H", bf16_bytes)]
    packed = bytearray(m * n // 2)
    scales = bytearray(m * n // 32)
    for row in range(m):
        fixed_line = tuple(_e3m2_fixed(code) for code in lines[row // 2])
        for group in range(n // 32):
            begin = row * n + group * 32
            block = values[begin:begin + 32]
            if any(not math.isfinite(value) for value in block):
                raise ValueError("FP6 output reference requires finite BF16 values")
            maximum = max(abs(value) for value in block)
            scale_code = (0 if maximum == 0 else
                          max(0, min(254, math.frexp(maximum)[1] - 1 + 127)))
            scale = math.ldexp(1.0, scale_code - 127)
            scales[row * (n // 32) + group] = scale_code
            for col, value in enumerate(block, group * 32):
                code = _e3m2_rne(_bf16_rne(value / scale))
                fixed = _e3m2_fixed(code)
                index = min(range(16), key=lambda i: abs(fixed - fixed_line[i]) & 0x1ff)
                packed[(row // 2) * n + col] |= index << (4 if row & 1 else 0)
    return bytes(packed), bytes(scales)


def _source_fp6_code(bits: int) -> int:
    """Radiance mx_fp_math.h BF16→E4M2→FP6 projection."""
    sign = -1.0 if bits & 0x8000 else 1.0
    exponent = (bits >> 7) & 255
    mantissa = bits & 127
    if exponent == 0:
        value = 0.0
    elif exponent == 255:
        raise ValueError("Radiance FP6 output requires finite BF16 values")
    else:
        unbiased = exponent - 127
        if -6 <= unbiased <= 7:
            high = (mantissa >> 5) & 3
            rounding = (mantissa >> 4) & 1
            sticky = (mantissa & 15) != 0
            rounded = high + (rounding & (sticky or bool(high & 1)))
            out_exponent = unbiased + int(rounded >= 4)
            value = (float("inf") if out_exponent > 7 else
                     (1.0 + (0 if rounded >= 4 else rounded) * 0.25) *
                     2.0 ** out_exponent)
        elif unbiased == -7:
            value = (2 if mantissa <= 32 else
                     3 if mantissa <= 95 else 4) / 256.0
        elif unbiased == -8:
            value = (1 if mantissa < 64 else 2) / 256.0
        elif unbiased == -9:
            value = (0 if mantissa == 0 else 1) / 256.0
        else:
            value = 0.0
    value *= sign
    magnitude = abs(value)
    if not math.isfinite(magnitude) or magnitude >= 32.0:
        value = sign * 28.0
    elif magnitude <= 0.0546875:
        value = 0.0
    elif 0.0625 <= magnitude <= 0.21875:
        value = sign * (0.0625 if magnitude <= 0.078125 else
                        0.125 if magnitude <= 0.15625 else 0.1875)
    if value == 0.0:
        return 0
    code_sign = 0x20 if value < 0 else 0
    magnitude = abs(value)
    if magnitude < 0.25:
        return code_sign | min(3, math.floor(magnitude / 0.0625 + 0.5))
    out_exponent = math.floor(math.log2(magnitude))
    base = 2.0 ** out_exponent
    output_mantissa = math.floor((magnitude - base) / (base / 4.0) + 0.5)
    if output_mantissa >= 4:
        output_mantissa = 0
        out_exponent += 1
    biased = min(7, out_exponent + 3)
    return code_sign | (biased << 2) | min(3, output_mantissa)


def quantize_bf16_radiance_header_fp6(
        bf16_bytes: bytes, m: int, n: int,
        output_lut_bytes: bytes) -> tuple[bytes, bytes]:
    """Reproduce Radiance's packed-M FP6 C projection and output scales."""
    if (m != 128 or n != 128 or len(bf16_bytes) != 2 * m * n or
            len(output_lut_bytes) != 64 * 12):
        raise ValueError("Radiance FP6 source projection requires its 128x128 LUT layout")
    lines = []
    for pair in range(64):
        packed = int.from_bytes(output_lut_bytes[pair * 12:(pair + 1) * 12], "little")
        lines.append(tuple((packed >> (6 * index)) & 63 for index in range(16)))
    values = [struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
              for (word,) in struct.iter_unpack("<H", bf16_bytes)]
    codes = bytearray(m * n // 2)
    scales = bytearray(m * n // 32)
    for row in range(m):
        fixed = tuple(_e3m2_fixed(code) for code in lines[row // 2])
        for group in range(n // 32):
            begin = row * n + group * 32
            block = values[begin:begin + 32]
            if any(not math.isfinite(value) for value in block):
                raise ValueError("Radiance FP6 output requires finite BF16 values")
            maximum = max(abs(value) for value in block)
            scale_code = (0 if maximum == 0 else max(
                0, min(254, math.floor(math.log2(maximum)) - 4 + 127)))
            scales[row * (n // 32) + group] = scale_code
            scale = math.ldexp(1.0, scale_code - 127)
            for col, value in enumerate(block, group * 32):
                bf16_bits = struct.unpack("<I", struct.pack("<f", value / scale))[0]
                bf16_bits = (bf16_bits + 0x7fff + ((bf16_bits >> 16) & 1)) >> 16
                input_code = _source_fp6_code(bf16_bits & 0xffff)
                quantized = _e3m2_fixed(input_code)
                index = min(range(16), key=lambda i: abs(quantized - fixed[i]) & 0x1ff)
                codes[(row // 2) * n + col] |= index << (4 if row & 1 else 0)
    return bytes(codes), bytes(scales)
