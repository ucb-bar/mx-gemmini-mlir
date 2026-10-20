"""Stream a selected BF16 VPU RMS sequence into its private FP8 destination.

The source endpoint must prove the original square/sum/divide/add/rsqrt and
two-multiply sequence, its private intermediates and the exact packed consumer.
This module emits that sequence; it does not recognize source model names.
"""

import struct
from dataclasses import dataclass

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect


@dataclass(frozen=True)
class RmsPackingKernel:
    source: str
    rows: int
    width: int
    rows_per_chunk: int
    packed_bytes: int


def emit_vpu_rms_packing(profile, *, rows, width, symbol):
    """Preserve BF16 arithmetic boundaries while removing intermediate DMA.

    Division by the admitted power-of-two width uses its exact BF16 reciprocal.
    Epsilon and gamma keep the original BF16 input rounding. A complete private
    FP8 code/scale tensor is published only after every streamed tile succeeds.
    """
    if (
        type(rows) is not int
        or rows < 16
        or rows % 16
        or type(width) is not int
        or not 32 <= width <= 4096
        or width & (width - 1)
        or rows * width > 0x7FFFFFFF
        or not symbol.isascii()
        or not symbol.isidentifier()
    ):
        raise ValueError("RMS packing needs mesh-aligned rows and a power-of-two width")
    if (
        profile.get("transport") != "rocket_rocc"
        or profile["geometry"]["mesh_columns"] != 16
    ):
        raise ValueError("RMS packing needs the Rocket DIM16 VPU")
    per_chunk = min(rows, 8192 // width)
    sizes = tuple(dict.fromkeys((per_chunk, rows % per_chunk)))
    reciprocal = struct.unpack("<I", struct.pack("<f", 1.0 / width))[0] >> 16
    issuers = []
    for count in sizes:
        if not count:
            continue
        vectors, rlen = count * width // 8, width // 8
        commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
        for buffer, address, length in (
            ("x", 0, vectors),
            ("gamma", 0x2000, vectors),
            ("epsilon", 0x3000, count),
        ):
            for start in range(0, length, 16):
                commands.append(
                    _transfer_rect(
                        2,
                        buffer,
                        start * 16,
                        address + start,
                        rows=min(16, length - start),
                    )
                )
        commands.extend(
            (
                vpu_command(
                    profile,
                    kind="mul",
                    src1_row=0,
                    src2_row=0,
                    dst_row=0x400,
                    rows=vectors,
                ),
                vpu_command(
                    profile,
                    kind="rsum",
                    src1_row=0x400,
                    src2_row=0,
                    dst_row=0x1000,
                    rows=vectors,
                    reduction_length=rlen,
                ),
                vpu_command(
                    profile,
                    kind="muls",
                    src1_row=0x1000,
                    src2_row=0,
                    dst_row=0x1000,
                    rows=count,
                    immediate_bf16=reciprocal,
                ),
                vpu_command(
                    profile,
                    kind="add",
                    src1_row=0x1000,
                    src2_row=0x3000,
                    dst_row=0x1000,
                    rows=count,
                ),
                vpu_command(
                    profile,
                    kind="rsqrt",
                    src1_row=0x1000,
                    src2_row=0,
                    dst_row=0x1000,
                    rows=count,
                ),
                vpu_command(
                    profile,
                    kind="mul",
                    src1_row=0,
                    src2_row=0x1000,
                    dst_row=0x400,
                    rows=vectors,
                    reduction_length=rlen,
                    broadcast=True,
                ),
                vpu_command(
                    profile,
                    kind="mul",
                    src1_row=0x400,
                    src2_row=0x2000,
                    dst_row=0x400,
                    rows=vectors,
                ),
                Fence(),
            )
        )
        for start in range(0, vectors, 16):
            commands.append(
                _transfer_rect(
                    3,
                    "result",
                    start * 16,
                    0x400 + start,
                    rows=min(16, vectors - start),
                )
            )
        commands.append(Fence())
        issuers.append(
            emit_c(
                commands,
                transport="rocket_rocc",
                buffers=("x", "gamma", "epsilon", "result"),
                symbol=f"{symbol}_issue_{count}",
                internal=True,
            )
        )
    cases = "\n".join(
        f"      case {count}: {symbol}_issue_{count}(x, g, e, y); break;"
        for count in sizes
        if count
    )
    source = f"""#include <stdint.h>
#include "runtime_quant.h"
{"".join(issuers)}
static uint16_t {symbol}_x[{per_chunk * width}] __attribute__((aligned(64)));
static uint16_t {symbol}_g[{per_chunk * width}] __attribute__((aligned(64)));
static uint16_t {symbol}_e[{per_chunk * 8}] __attribute__((aligned(64)));
static uint16_t {symbol}_y[{per_chunk * width}] __attribute__((aligned(64)));
static uint16_t {symbol}_narrow(float x) {{
  union {{float f; uint32_t u;}} w = {{.f=x}};
  if ((w.u & 0x7f800000u)==0x7f800000u && (w.u & 0x7fffffu)) __builtin_trap();
  w.u += 0x7fffu + ((w.u >> 16) & 1u);
  return (uint16_t)(w.u >> 16);
}}
void {symbol}(const float *input, const float *gamma, const float *divisor,
              const float *epsilon, uint8_t *packed) {{
  if (!input || !gamma || !divisor || !epsilon || !packed || *divisor != {width}.0f)
    __builtin_trap();
  uint16_t eps = {symbol}_narrow(*epsilon);
  for (uint32_t col=0; col<{width}; ++col) {{
    uint16_t word = {symbol}_narrow(gamma[col]);
    for (uint32_t row=0; row<{per_chunk}; ++row) {symbol}_g[row*{width}+col]=word;
  }}
  for (uint32_t i=0; i<{per_chunk * 8}; ++i) {symbol}_e[i]=eps;
  for (uint32_t base=0; base<{rows}; base+={per_chunk}) {{
    uint32_t count={rows}-base;
    if (count>{per_chunk}) count={per_chunk};
    for (uint32_t i=0; i<count*{width}; ++i)
      {symbol}_x[i]={symbol}_narrow(input[base*{width}+i]);
    const void *x={symbol}_x, *g={symbol}_g, *e={symbol}_e, *y={symbol}_y;
    switch(count) {{
{cases}
      default: __builtin_trap();
    }}
    if (mx_pack_activation_bf16_rows({symbol}_y,{rows},{width},base,count,
                                    packed,packed+{rows * width}u)) __builtin_trap();
  }}
}}
"""
    return RmsPackingKernel(source, rows, width, per_chunk, rows * width * 33 // 32)
