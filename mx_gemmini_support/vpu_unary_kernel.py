"""Contiguous BF16 VPU unary kernels for source tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect


@dataclass(frozen=True)
class UnaryKernel:
    source: str
    elements: int
    elements_per_chunk: int
    operation: str


def sigmoid_steps(*, source_row=0, temporary_row=0x1000, result_row=0x2000):
    """The existing stepwise BF16 sigmoid policy, shared by resident emitters."""
    return (
        {"kind": "muls", "src1_row": source_row, "src2_row": 0,
         "dst_row": temporary_row, "immediate_bf16": 0xbf80},
        {"kind": "exp", "src1_row": temporary_row, "src2_row": 0, "dst_row": temporary_row},
        {"kind": "adds", "src1_row": temporary_row, "src2_row": 0,
         "dst_row": temporary_row, "immediate_bf16": 0x3f80},
        {"kind": "rcp", "src1_row": temporary_row, "src2_row": 0, "dst_row": result_row},
    )


def _issue(profile: dict, *, elements: int, operation: str,
           symbol: str) -> str:
    rows = (elements + 7) // 8
    if rows > 1024:
        raise ValueError("VPU unary tile exceeds its scratchpad window")
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for start in range(0, rows, 16):
        commands.append(_transfer_rect(2, "input", start * 16, start,
                                       rows=min(16, rows - start)))
    if operation == "sigmoid":
        commands.extend(vpu_command(profile, rows=rows, **step) for step in sigmoid_steps())
        commands.append(Fence())
    elif operation == "tanh":
        # tanh(x) = 2 / (1 + exp(-2*x)) - 1; round each VPU step to BF16.
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0xc000),
            vpu_command(profile, kind="exp", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="adds", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0x3f80),
            vpu_command(profile, kind="rcp", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="muls", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0x4000),
            vpu_command(profile, kind="adds", src1_row=0x1000, src2_row=0,
                        dst_row=0x2000, rows=rows, immediate_bf16=0xbf80), Fence(),
        ))
    elif operation == "gelu":
        # x / (1 + exp(-2*sqrt(2/pi)*(x + 0.044715*x^3))).
        # Every intermediate is BF16, as in the selected hardware VPU.
        commands.extend((
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="mul", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="muls", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0x3d37),
            vpu_command(profile, kind="add", src1_row=0, src2_row=0x1000,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="muls", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0xbfcc),
            vpu_command(profile, kind="exp", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="adds", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0x3f80),
            vpu_command(profile, kind="rcp", src1_row=0x1000, src2_row=0,
                        dst_row=0x1000, rows=rows),
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0x1000,
                        dst_row=0x2000, rows=rows), Fence(),
        ))
    elif operation == "pow10000":
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=0x1000, rows=rows, immediate_bf16=0x4113),
            vpu_command(profile, kind="exp", src1_row=0x1000, src2_row=0,
                        dst_row=0x2000, rows=rows), Fence(),
        ))
    elif operation == "neg":
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=0x2000, rows=rows, immediate_bf16=0xbf80),
            Fence(),
        ))
    elif operation == "square":
        commands.extend((
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0,
                        dst_row=0x2000, rows=rows), Fence(),
        ))
    else:
        commands.extend((
            vpu_command(profile, kind=operation, src1_row=0, src2_row=0,
                        dst_row=0x2000, rows=rows), Fence(),
        ))
    for start in range(0, rows, 16):
        commands.append(_transfer_rect(3, "result", start * 16,
                                       0x2000 + start,
                                       rows=min(16, rows - start)))
    commands.append(Fence())
    return emit_c(commands, transport="rocket_rocc",
                  buffers=("input", "result"), symbol=symbol, internal=True)


def emit_vpu_unary(profile: dict, *, shape: tuple[int, ...], dtype: str,
                   operation: str, symbol: str) -> UnaryKernel:
    """Emit BF16 unary operations with explicit f32 conversion."""
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu")):
        raise ValueError("VPU unary needs the selected DIM16 Rocket MX VPU")
    if (not 1 <= len(shape) <= 4 or
            any(type(dim) is not int or dim <= 0 for dim in shape) or
            dtype not in {"bf16", "f32"} or operation not in {"rsqrt", "sigmoid", "tanh", "square", "neg", "gelu", "pow10000"} or
            not symbol.isidentifier() or not symbol.isascii()):
        raise ValueError("VPU unary needs a static tensor and supported operation")
    elements = prod(shape)
    chunk = min(elements, 8192)
    tail = elements % chunk
    sizes = (chunk,) if not tail else (chunk, tail)
    issuers = "\n".join(_issue(profile, elements=count, operation=operation,
                                symbol=f"{symbol}_issue_{count}")
                        for count in sizes)
    dispatch = "\n".join(
        f"    case {count}: {symbol}_issue_{count}(a, c); break;"
        for count in sizes)
    scalar = "float" if dtype == "f32" else "uint16_t"
    input_read = (f"{symbol}_narrow(input[base + index])" if dtype == "f32"
                  else "input[base + index]")
    output_read = (f"{symbol}_widen({symbol}_c[index])" if dtype == "f32"
                   else f"{symbol}_c[index]")
    source = f'''#include <stdint.h>
{issuers}
static uint16_t {symbol}_a[{(chunk + 7) // 8 * 8}] __attribute__((aligned(64)));
static uint16_t {symbol}_c[{(chunk + 7) // 8 * 8}] __attribute__((aligned(64)));
static uint16_t {symbol}_narrow(float value) {{
  union {{float f; uint32_t u;}} word;
  word.f = value;
  if ((word.u & 0x7f800000u) == 0x7f800000u &&
      (word.u & 0x007fffffu)) __builtin_trap();
  word.u += 0x7fffu + ((word.u >> 16) & 1u);
  return (uint16_t)(word.u >> 16);
}}
static float {symbol}_widen(uint16_t value) {{
  union {{float f; uint32_t u;}} word;
  word.u = (uint32_t)value << 16;
  return word.f;
}}
void {symbol}(const {scalar} *input, {scalar} *output) {{
  if (!input || !output) __builtin_trap();
  for (uint32_t base = 0; base < {elements}; base += {chunk}) {{
    uint32_t count = {elements} - base;
    if (count > {chunk}) count = {chunk};
    for (uint32_t index = 0; index < (count + 7u) / 8u * 8u; ++index)
      {symbol}_a[index] = index < count ? {input_read} : 1;
    const void *a = {symbol}_a, *c = {symbol}_c;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t index = 0; index < count; ++index)
      output[base + index] = {output_read};
  }}
}}
'''
    return UnaryKernel(source, elements, chunk, operation)
