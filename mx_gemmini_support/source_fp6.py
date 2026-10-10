"""Read Radiance FP6 GEMM's packed source operands and three LUT banks.

This binds the handwritten fullout kernel and source-derived requant fixtures.
It checks packing and LUT interpretation before physical lowering; the
model2MLIR capture alone is not a numerical result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .fp6 import pack_fp6_indexed_contraction
from .layout import pack_fp6_lut
from .source_gemm import SourceGemm


@dataclass(frozen=True)
class SourceFp6Payload:
    header_sha256: str
    activation_lut_line0: tuple[int, ...]
    weight_lut_line0: tuple[int, ...]
    activation_bytes: bytes
    weight_bytes: bytes
    activation_lut_bytes: bytes
    weight_lut_bytes: bytes
    output_lut_bytes: bytes
    activation_scale_bytes: bytes
    weight_scale_bytes: bytes
    golden_bf16_bytes: bytes

    def digests(self) -> dict[str, str]:
        return {
            name + "_sha256": hashlib.sha256(getattr(self, name)).hexdigest()
            for name in ("activation_bytes", "weight_bytes", "activation_lut_bytes",
                         "weight_lut_bytes", "output_lut_bytes", "activation_scale_bytes",
                         "weight_scale_bytes", "golden_bf16_bytes")
        }


def _array(source: str, *, name: str, ctype: str, dimensions: str,
           count: int, maximum: int) -> tuple[int, ...]:
    declaration = rf"static const {ctype} {name}{re.escape(dimensions)}\s*=\s*\{{(.*?)\n\}};"
    found = re.search(declaration, source, re.DOTALL)
    if found is None:
        raise ValueError(f"source header lacks expected {name} declaration")
    body = found.group(1)
    # The checked-in header uses integer literals only. Reject expressions,
    # identifiers, and malformed separators instead of interpreting C here.
    values = re.findall(r"0[xX][0-9a-fA-F]+|\d+", body)
    residue = re.sub(r"0[xX][0-9a-fA-F]+|\d+|[{},\s]", "", body)
    if residue or len(values) != count:
        raise ValueError(f"source {name} contains nonliteral or wrong-sized data")
    decoded = tuple(int(value, 0) for value in values)
    if any(value < 0 or value > maximum for value in decoded):
        raise ValueError(f"source {name} exceeds its declared element width")
    return decoded


def _bytes(values: tuple[int, ...], width: int) -> bytes:
    return b"".join(value.to_bytes(width, "little") for value in values)


def _lut_lines(words: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    lines = []
    for start in range(0, len(words), 3):
        packed = int.from_bytes(_bytes(words[start:start + 3], 4), "little")
        lines.append(tuple((packed >> (6 * index)) & 0x3f for index in range(16)))
    return tuple(lines)


def read_source_fp6_payload(kernel: SourceGemm) -> SourceFp6Payload:
    """Prove that the source's nibble indices and row LUTs round-trip exactly."""
    valid = ((not kernel.quant_output and kernel.shape == (128, 128, 2048) and
              kernel.tile == (128, 128, 128)) or
             (kernel.quant_output and kernel.shape in {
                 (128, 128, 128), (128, 128, 512)} and
              kernel.tile == kernel.shape))
    if (kernel.datatype != "FP6" or kernel.acc_to_gmem or not valid or
            not kernel.data_header_present):
        raise ValueError("source FP6 payload requires a checked fullout or generated requant driver")
    header = kernel.data_header.read_bytes()
    source = header.decode("ascii")
    m, n, k = kernel.shape
    a = _array(source, name="A_in_hw", ctype="uint8_t", dimensions=f"[64][{k}]",
               count=m * k // 2, maximum=255)
    b = _array(source, name="B_in", ctype="uint8_t",
               dimensions="[MATMUL_K][MATMUL_N / 2]", count=k * n // 2, maximum=255)
    lut_words = {
        name: _array(source, name=name, ctype="uint32_t", dimensions="[64][3]",
                     count=64 * 3, maximum=0xffffffff)
        for name in ("A_lut", "B_lut", "C_lut")
    }
    lut_lines = {name: _lut_lines(words) for name, words in lut_words.items()}
    for name, lines in lut_lines.items():
        if pack_fp6_lut(lines) != _bytes(lut_words[name], 4):
            raise ValueError(f"source FP6 {name} does not match the selected RTL LUT packing")

    a_codes = [[0] * k for _ in range(m)]
    for pair in range(m // 2):
        line = lut_lines["A_lut"][pair]
        for inner in range(k):
            value = a[pair * k + inner]
            a_codes[2 * pair][inner] = line[value & 0xf]
            a_codes[2 * pair + 1][inner] = line[value >> 4]
    b_codes = [[0] * n for _ in range(k)]
    for inner in range(k):
        for pair in range(n // 2):
            value = b[inner * (n // 2) + pair]
            line = lut_lines["B_lut"][pair]
            b_codes[inner][2 * pair] = line[value & 0xf]
            b_codes[inner][2 * pair + 1] = line[value >> 4]
    repacked = pack_fp6_indexed_contraction(
        a_codes, b_codes, activation_lut=lut_lines["A_lut"],
        weight_lut=lut_lines["B_lut"], granularity_shift=1)
    if repacked.activation_bytes != bytes(a) or repacked.weight_bytes != bytes(b):
        raise ValueError("source FP6 indexed operand bytes do not round-trip")

    a_scales = _array(source, name="A_scales_row", ctype="uint8_t",
                      dimensions="[MATMUL_GK][MATMUL_M]", count=k // 32 * m,
                      maximum=255)
    b_scales = _array(source, name="B_scales_col", ctype="uint8_t",
                      dimensions="[MATMUL_GK][MATMUL_N]", count=k // 32 * n,
                      maximum=255)
    golden = _array(source, name="C_out_bf16", ctype="uint16_t",
                    dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                    maximum=0xffff)
    return SourceFp6Payload(
        hashlib.sha256(header).hexdigest(), lut_lines["A_lut"][0],
        lut_lines["B_lut"][0], bytes(a), bytes(b),
        repacked.activation_lut_bytes, repacked.weight_lut_bytes,
        pack_fp6_lut(lut_lines["C_lut"]), bytes(a_scales), bytes(b_scales),
        _bytes(golden, 2),
    )
