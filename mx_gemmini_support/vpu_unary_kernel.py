"""Contiguous BF16 VPU unary kernels for source tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect
from .target_profile import model_vpu_lanes


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
    lanes = model_vpu_lanes(profile)
    assert lanes is not None
    row_bytes = lanes * 2
    rows = (elements + lanes - 1) // lanes
    if rows > 1024:
        raise ValueError("VPU unary tile exceeds its scratchpad window")
    total_rows = profile["resources"]["scratchpad_bytes"] // row_bytes
    temporary_row, result_row = ((0x1000, 0x2000) if total_rows >= 0x2400
                                 else (0x400, 0x800))
    if result_row + rows > total_rows:
        raise ValueError("VPU unary scratchpad windows exceed selected profile")
    commands = [_cmd(7, 0, 0), _config_ld(row_bytes, dim=row_bytes),
                _config_st(row_bytes)]
    for start in range(0, rows, 16):
        commands.append(_transfer_rect(2, "input", start * row_bytes, start,
                                       rows=min(16, rows - start), cols=row_bytes))
    if operation == "sigmoid":
        commands.extend(vpu_command(profile, rows=rows, **step) for step in
                        sigmoid_steps(temporary_row=temporary_row,
                                      result_row=result_row))
        commands.append(Fence())
    elif operation == "tanh":
        # tanh(x) = 2 / (1 + exp(-2*x)) - 1; round each VPU step to BF16.
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0xc000),
            vpu_command(profile, kind="exp", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="adds", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0x3f80),
            vpu_command(profile, kind="rcp", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="muls", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0x4000),
            vpu_command(profile, kind="adds", src1_row=temporary_row, src2_row=0,
                        dst_row=result_row, rows=rows, immediate_bf16=0xbf80), Fence(),
        ))
    elif operation == "gelu":
        # x / (1 + exp(-2*sqrt(2/pi)*(x + 0.044715*x^3))).
        # Every intermediate is BF16, as in the selected hardware VPU.
        commands.extend((
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="mul", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="muls", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0x3d37),
            vpu_command(profile, kind="add", src1_row=0, src2_row=temporary_row,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="muls", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0xbfcc),
            vpu_command(profile, kind="exp", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="adds", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0x3f80),
            vpu_command(profile, kind="rcp", src1_row=temporary_row, src2_row=0,
                        dst_row=temporary_row, rows=rows),
            vpu_command(profile, kind="mul", src1_row=0, src2_row=temporary_row,
                        dst_row=result_row, rows=rows), Fence(),
        ))
    elif operation == "pow10000":
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=temporary_row, rows=rows, immediate_bf16=0x4113),
            vpu_command(profile, kind="exp", src1_row=temporary_row, src2_row=0,
                        dst_row=result_row, rows=rows), Fence(),
        ))
    elif operation == "neg":
        commands.extend((
            vpu_command(profile, kind="muls", src1_row=0, src2_row=0,
                        dst_row=result_row, rows=rows, immediate_bf16=0xbf80),
            Fence(),
        ))
    elif operation == "square":
        commands.extend((
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0,
                        dst_row=result_row, rows=rows), Fence(),
        ))
    else:
        commands.extend((
            vpu_command(profile, kind=operation, src1_row=0, src2_row=0,
                        dst_row=result_row, rows=rows), Fence(),
        ))
    for start in range(0, rows, 16):
        commands.append(_transfer_rect(3, "result", start * row_bytes,
                                       result_row + start,
                                       rows=min(16, rows - start), cols=row_bytes))
    commands.append(Fence())
    return emit_c(commands, transport="rocket_rocc",
                  buffers=("input", "result"), symbol=symbol, internal=True)


def emit_vpu_unary(profile: dict, *, shape: tuple[int, ...], dtype: str,
                   operation: str, symbol: str,
                   sigmoid_implementation: str = "bf16") -> UnaryKernel:
    """Emit BF16 unary operations with explicit f32 conversion."""
    lanes = model_vpu_lanes(profile)
    if lanes is None:
        raise ValueError("VPU unary needs a selected Rocket MX VPU")
    if (not 1 <= len(shape) <= 4 or
            any(type(dim) is not int or dim <= 0 for dim in shape) or
            dtype not in {"bf16", "f32"} or operation not in {"rsqrt", "sigmoid", "tanh", "square", "neg", "gelu", "pow10000"} or
            not symbol.isidentifier() or not symbol.isascii()):
        raise ValueError("VPU unary needs a static tensor and supported operation")
    if sigmoid_implementation not in {"bf16", "f32_reciprocal"}:
        raise ValueError("unknown sigmoid arithmetic implementation")
    if sigmoid_implementation == "f32_reciprocal":
        if operation != "sigmoid" or dtype != "f32":
            raise ValueError("hybrid sigmoid requires a selected FP32 sigmoid boundary")
        return _emit_hybrid_sigmoid(profile,shape,symbol)
    elements = prod(shape)
    chunk = min(elements, 1024 * lanes)
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
static uint16_t {symbol}_a[{(chunk + lanes - 1) // lanes * lanes}] __attribute__((aligned(64)));
static uint16_t {symbol}_c[{(chunk + lanes - 1) // lanes * lanes}] __attribute__((aligned(64)));
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
    for (uint32_t index = 0; index < (count + {lanes - 1}u) / {lanes}u * {lanes}u; ++index)
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


def _emit_hybrid_sigmoid(profile,shape,symbol):
    """Use BF16 EXP with FP32 negation, addition and reciprocal.

    This is an explicit numerical alternative. The endpoint retains the FP32
    output of 1/(1+exp(-x)); neighboring operators keep their own selected policy.
    """
    elements=prod(shape)
    if elements>0x7fffffff:
        raise ValueError("hybrid sigmoid exceeds the selected runtime extent")
    lanes=model_vpu_lanes(profile)
    assert lanes is not None
    chunk=min(elements,1024*lanes)
    sizes=tuple(dict.fromkeys((chunk,elements%chunk)))
    issuers="\n".join(_issue(profile,elements=count,operation="exp",symbol=f"{symbol}_issue_{count}")
                        for count in sizes if count)
    dispatch="\n".join(f"      case {count}: {symbol}_issue_{count}(a,c); break;"
                         for count in sizes if count)
    source=f'''#include <stdint.h>
{issuers}
static uint16_t {symbol}_input[{(chunk+lanes-1)//lanes*lanes}] __attribute__((aligned(64)));
static uint16_t {symbol}_exp[{(chunk+lanes-1)//lanes*lanes}] __attribute__((aligned(64)));
void {symbol}(const float *input,float *output) {{
  if (!input || !output) __builtin_trap();
  for (uint32_t base=0; base<{elements}u; base+={chunk}u) {{
    uint32_t count={elements}u-base;
    if (count>{chunk}u) count={chunk}u;
    for (uint32_t i=0; i<(count+{lanes-1}u)/{lanes}u*{lanes}u; ++i) {{
      union {{float f;uint32_t u;}} word={{.f=i<count ? -input[base+i] : 0.0f}};
      if ((word.u&0x7f800000u)==0x7f800000u && (word.u&0x007fffffu)) __builtin_trap();
      word.u+=0x7fffu+((word.u>>16)&1u);
      {symbol}_input[i]=word.u>>16;
    }}
    const void *a={symbol}_input,*c={symbol}_exp;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t i=0; i<count; ++i) {{
      union {{float f;uint32_t u;}} word={{.u=(uint32_t){symbol}_exp[i]<<16}};
      output[base+i]=1.0f/(1.0f+word.f);
    }}
  }}
}}
'''
    return UnaryKernel(source,elements,chunk,"sigmoid")
