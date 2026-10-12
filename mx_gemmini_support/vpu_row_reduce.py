"""BF16 VPU row sum/max with explicit f32 input and output boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect


@dataclass(frozen=True)
class RowReduceKernel:
    source: str
    logical_rows: int
    row_width: int
    padded_width: int
    rows_per_chunk: int
    operation: str


def _issuer(profile: dict, *, rows: int, rlen: int, operation: str,
            symbol: str) -> str:
    spad_rows = rows * rlen
    if spad_rows > 1024:
        raise ValueError("VPU reduction tile exceeds its scratchpad window")
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for start in range(0, spad_rows, 16):
        commands.append(_transfer_rect(2, "input", start * 16, start,
                                       rows=min(16, spad_rows - start)))
    commands.extend((
        vpu_command(profile, kind=operation, src1_row=0, src2_row=0,
                    dst_row=0x2000, rows=spad_rows, reduction_length=rlen),
        Fence(),
    ))
    for start in range(0, rows, 16):
        commands.append(_transfer_rect(3, "result", start * 16,
                                       0x2000 + start,
                                       rows=min(16, rows - start)))
    commands.append(Fence())
    return emit_c(commands, transport="rocket_rocc",
                  buffers=("input", "result"), symbol=symbol, internal=True)


def emit_vpu_row_reduce(profile: dict, *, logical_rows: int, row_width: int,
                        operation: str, symbol: str) -> RowReduceKernel:
    """Reduce the last axis of a contiguous f32 tensor to one f32 per row."""
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu")):
        raise ValueError("row reduction needs the selected DIM16 Rocket MX VPU")
    if (type(logical_rows) is not int or logical_rows <= 0 or
            type(row_width) is not int or not 1 <= row_width <= 8184 or
            operation not in {"rsum", "rmax"} or
            not symbol.isidentifier() or not symbol.isascii()):
        raise ValueError("row reduction needs static extents and known operation")
    padded = (row_width + 7) // 8 * 8
    rlen = padded // 8
    rows_per_chunk = min(logical_rows, 1024 // rlen)
    if not rows_per_chunk:
        raise ValueError("row reduction cannot fit selected VPU scratchpad")
    tail = logical_rows % rows_per_chunk
    sizes = (rows_per_chunk,) if not tail else (rows_per_chunk, tail)
    issuers = "\n".join(_issuer(profile, rows=count, rlen=rlen,
                                operation=operation,
                                symbol=f"{symbol}_issue_{count}")
                        for count in sizes)
    dispatch = "\n".join(
        f"    case {count}: {symbol}_issue_{count}(input, result); break;"
        for count in sizes)
    pad = "0u" if operation == "rsum" else "0xff80u"
    source = f'''#include <stdint.h>
{issuers}
static uint16_t {symbol}_input[{rows_per_chunk * padded}] __attribute__((aligned(64)));
static uint16_t {symbol}_result[{rows_per_chunk * 8}] __attribute__((aligned(64)));
void {symbol}(const float *source, float *destination) {{
  if (!source || !destination) __builtin_trap();
  for (uint32_t base = 0; base < {logical_rows}; base += {rows_per_chunk}) {{
    uint32_t count = {logical_rows} - base;
    if (count > {rows_per_chunk}) count = {rows_per_chunk};
    for (uint32_t row=0; row<count; ++row)
      for (uint32_t col=0; col<{padded}; ++col) {{
        if (col >= {row_width}) {{ {symbol}_input[row*{padded}+col]={pad}; continue; }}
        union {{ float f; uint32_t u; }} word;
        word.f=source[(base+row)*{row_width}+col];
        if ((word.u & 0x7f800000u) == 0x7f800000u &&
            (word.u & 0x007fffffu)) __builtin_trap();
        word.u += 0x7fffu + ((word.u >> 16) & 1u);
        {symbol}_input[row*{padded}+col]=(uint16_t)(word.u>>16);
      }}
    const void *input={symbol}_input, *result={symbol}_result;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t row=0; row<count; ++row) {{
      union {{ float f; uint32_t u; }} word;
      word.u=(uint32_t){symbol}_result[row*8] << 16;
      destination[base+row]=word.f;
    }}
  }}
}}
'''
    return RowReduceKernel(source, logical_rows, row_width, padded,
                           rows_per_chunk, operation)
