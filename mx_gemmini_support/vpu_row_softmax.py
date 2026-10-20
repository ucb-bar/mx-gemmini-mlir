"""Profile-checked softmax alternatives at explicit F32 tensor boundaries.

The default rounds scores and performs the complete softmax on the BF16 VPU.
The optional hybrid keeps FP32 normalization and batches BF16 exponentials.
Each mode declares its rounding boundary and requires numerical qualification.
"""

from __future__ import annotations

from dataclasses import dataclass

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect


@dataclass(frozen=True)
class RowSoftmaxKernel:
    source: str
    logical_rows: int
    row_width: int
    padded_width: int
    rows_per_chunk: int
    vpu_steps: tuple[str, ...]


def _issuer(profile: dict, *, rows: int, rlen: int, symbol: str) -> str:
    spad_rows = rows * rlen
    if spad_rows > 1024:
        raise ValueError("softmax score tile overlaps its probability scratchpad")
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for start in range(0, spad_rows, 16):
        count = min(16, spad_rows - start)
        commands.append(_transfer_rect(2, "score", start * 16, start,
                                       rows=count))
    commands.extend((
        vpu_command(profile, kind="rmax", src1_row=0, src2_row=0,
                    dst_row=0x1000, rows=spad_rows, reduction_length=rlen),
        vpu_command(profile, kind="sub", src1_row=0, src2_row=0x1000,
                    dst_row=0x2000, rows=spad_rows, reduction_length=rlen,
                    broadcast=True),
        vpu_command(profile, kind="exp", src1_row=0x2000, src2_row=0,
                    dst_row=0x2000, rows=spad_rows),
        vpu_command(profile, kind="rsum", src1_row=0x2000, src2_row=0,
                    dst_row=0x3000, rows=spad_rows, reduction_length=rlen),
        vpu_command(profile, kind="rcp", src1_row=0x3000, src2_row=0,
                    dst_row=0x3000, rows=rows),
        vpu_command(profile, kind="mul", src1_row=0x2000, src2_row=0x3000,
                    dst_row=0x0400, rows=spad_rows, reduction_length=rlen,
                    broadcast=True),
        Fence(),
    ))
    for start in range(0, spad_rows, 16):
        count = min(16, spad_rows - start)
        commands.append(_transfer_rect(3, "result", start * 16,
                                       0x0400 + start, rows=count))
    commands.append(Fence())
    return emit_c(commands, transport="rocket_rocc",
                  buffers=("score", "result"), symbol=symbol, internal=True)


def emit_vpu_row_softmax(profile: dict, *, logical_rows: int,
                         row_width: int, symbol: str,
                         implementation: str = "bf16") -> RowSoftmaxKernel:
    """Emit a freestanding f32-pointer softmax with explicit BF16 rounding.

    The function is compiled for static source extents and processes arbitrary
    row counts in disjoint tiles. It is a target kernel, not a host graph
    rewrite: a caller must still bind a source softmax to this function.
    """
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu")):
        raise ValueError("row softmax needs the selected DIM16 Rocket MX VPU")
    if (type(logical_rows) is not int or logical_rows <= 0 or
            type(row_width) is not int or row_width <= 0 or row_width > 8184 or
            not symbol.isidentifier() or not symbol.isascii()):
        raise ValueError("row softmax needs positive static extents and a C symbol")
    if implementation not in {"bf16", "f32_reduce"}:
        raise ValueError("unknown row softmax implementation")
    if implementation == "f32_reduce":
        return _emit_exp_softmax(profile, logical_rows, row_width, symbol)
    padded = (row_width + 7) // 8 * 8
    rlen = padded // 8
    if rlen > 1023:
        raise ValueError("softmax reduction exceeds VPU encoding")
    rows_per_chunk = min(logical_rows, 1024 // rlen)
    if not rows_per_chunk:
        raise ValueError("softmax row cannot fit the VPU scratchpad")
    tail = logical_rows % rows_per_chunk
    sizes = (rows_per_chunk,) if not tail else (rows_per_chunk, tail)
    issuers = "\n".join(_issuer(profile, rows=count, rlen=rlen,
                                  symbol=f"{symbol}_issue_{count}")
                        for count in sizes)
    dispatch = "\n".join(
        f"    case {count}: {symbol}_issue_{count}(score, result); break;"
        for count in sizes)
    source = f'''#include <stdint.h>

{issuers}

static uint16_t {symbol}_score[{rows_per_chunk * padded}] __attribute__((aligned(64)));
static uint16_t {symbol}_result[{rows_per_chunk * padded}] __attribute__((aligned(64)));

void {symbol}(const float *input, float *output) {{
  if (!input || !output) __builtin_trap();
  uintptr_t a = (uintptr_t)input, b = (uintptr_t)output;
  uintptr_t bytes = (uintptr_t){logical_rows} * {row_width} * sizeof(float);
  if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes ||
      !(a + bytes <= b || b + bytes <= a)) __builtin_trap();
  for (uint32_t base = 0; base < {logical_rows}; base += {rows_per_chunk}) {{
    uint32_t count = {logical_rows} - base;
    if (count > {rows_per_chunk}) count = {rows_per_chunk};
    for (uint32_t row = 0; row < count; ++row) {{
      for (uint32_t col = 0; col < {padded}; ++col) {{
        if (col >= {row_width}) {{ {symbol}_score[row * {padded} + col] = 0xff80u; continue; }}
        union {{ float value; uint32_t bits; }} word;
        word.value = input[(base + row) * {row_width} + col];
        if ((word.bits & 0x7f800000u) == 0x7f800000u &&
            (word.bits != 0xff800000u)) __builtin_trap();
        word.bits += 0x7fffu + ((word.bits >> 16) & 1u);
        {symbol}_score[row * {padded} + col] = (uint16_t)(word.bits >> 16);
      }}
    }}
    const void *score = {symbol}_score;
    const void *result = {symbol}_result;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t row = 0; row < count; ++row)
      for (uint32_t col = 0; col < {row_width}; ++col) {{
        union {{ float value; uint32_t bits; }} word;
        word.bits = (uint32_t){symbol}_result[row * {padded} + col] << 16;
        output[(base + row) * {row_width} + col] = word.value;
      }}
  }}
}}
'''
    return RowSoftmaxKernel(source, logical_rows, row_width, padded,
                            rows_per_chunk,
                            ("rmax", "sub", "exp", "rsum", "rcp", "mul"))


def _emit_exp_softmax(profile, logical_rows, row_width, symbol):
    """Batch vector exponentials while retaining FP32 max, sum and division.

    This is an explicit numerical alternative to the entirely BF16 VPU
    softmax. Only the shifted exponential input and its result round to BF16.
    Rows remain independent; masked negative infinities contribute exact zero.
    """
    padded = (row_width + 7) // 8 * 8
    per_chunk = min(logical_rows, 8192 // padded)
    sizes = tuple(dict.fromkeys((per_chunk, logical_rows % per_chunk)))
    issuers = []
    for count in sizes:
        if not count:
            continue
        rows = count * padded // 8
        commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
        for start in range(0, rows, 16):
            commands.append(_transfer_rect(2, "score", start * 16, start,
                                            rows=min(16, rows - start)))
        commands.extend((vpu_command(profile, kind="exp", src1_row=0, src2_row=0,
                                      dst_row=0x2000, rows=rows), Fence()))
        for start in range(0, rows, 16):
            commands.append(_transfer_rect(3, "result", start * 16, 0x2000 + start,
                                            rows=min(16, rows - start)))
        commands.append(Fence())
        issuers.append(emit_c(commands, transport="rocket_rocc", buffers=("score", "result"),
                             symbol=f"{symbol}_exp_{count}", internal=True))
    dispatch = "\n".join(f"      case {count}: {symbol}_exp_{count}(score, result); break;"
                         for count in sizes if count)
    source = f'''#include <stdint.h>
#include <math.h>
{''.join(issuers)}
static uint16_t {symbol}_score[{per_chunk * padded}] __attribute__((aligned(64)));
static uint16_t {symbol}_result[{per_chunk * padded}] __attribute__((aligned(64)));
void {symbol}(const float *input, float *output) {{
  if (!input || !output) __builtin_trap();
  uintptr_t a = (uintptr_t)input, b = (uintptr_t)output;
  uintptr_t bytes = (uintptr_t){logical_rows} * {row_width} * sizeof(float);
  if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes ||
      !(a + bytes <= b || b + bytes <= a)) __builtin_trap();
  for (uint32_t base = 0; base < {logical_rows}; base += {per_chunk}) {{
    uint32_t count = {logical_rows} - base;
    if (count > {per_chunk}) count = {per_chunk};
    for (uint32_t row = 0; row < count; ++row) {{
      const float *in = input + (base + row) * {row_width};
      float maximum = -INFINITY;
      for (uint32_t col = 0; col < {row_width}; ++col) {{
        union {{ float f; uint32_t u; }} x = {{ .f = in[col] }};
        if ((x.u & 0x7f800000u) == 0x7f800000u && x.u != 0xff800000u) __builtin_trap();
        if (x.f > maximum) maximum = x.f;
      }}
      if (maximum == -INFINITY) __builtin_trap();
      for (uint32_t col = 0; col < {padded}; ++col) {{
        union {{ float f; uint32_t u; }} x;
        x.f = col < {row_width} ? in[col] - maximum : -INFINITY;
        x.u += 0x7fffu + ((x.u >> 16) & 1u);
        {symbol}_score[row * {padded} + col] = (uint16_t)(x.u >> 16);
      }}
    }}
    const void *score = {symbol}_score, *result = {symbol}_result;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t row = 0; row < count; ++row) {{
      float sum = 0.0f;
      for (uint32_t col = 0; col < {row_width}; ++col) {{
        union {{ float f; uint32_t u; }} x;
        x.u = (uint32_t){symbol}_result[row * {padded} + col] << 16;
        sum += x.f;
      }}
      for (uint32_t col = 0; col < {row_width}; ++col) {{
        union {{ float f; uint32_t u; }} x;
        x.u = (uint32_t){symbol}_result[row * {padded} + col] << 16;
        output[(base + row) * {row_width} + col] = x.f / sum;
      }}
    }}
  }}
}}
'''
    return RowSoftmaxKernel(source, logical_rows, row_width, padded, per_chunk, ("exp",))
