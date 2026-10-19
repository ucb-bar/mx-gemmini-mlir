"""Profile-checked MX VPU binary arithmetic with exact tensor broadcasts."""

from __future__ import annotations

from dataclasses import dataclass

from .command_ir import Fence, emit_c, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect


@dataclass(frozen=True)
class AddKernel:
    source: str
    elements: int
    elements_per_chunk: int
    dtype: str
    broadcast: str


def _issue(profile: dict, *, elements: int, symbol: str, operation: str) -> str:
    spad_rows = (elements + 7) // 8
    if spad_rows > 1024:
        raise ValueError("VPU add tile exceeds its scratchpad windows")
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for start in range(0, spad_rows, 16):
        commands.append(_transfer_rect(2, "lhs", start * 16, start,
                                       rows=min(16, spad_rows - start)))
    for start in range(0, spad_rows, 16):
        commands.append(_transfer_rect(2, "rhs", start * 16, 0x1000 + start,
                                       rows=min(16, spad_rows - start)))
    if operation == "div":
        commands.extend((
            vpu_command(profile, kind="rcp", src1_row=0x1000, src2_row=0,
                        dst_row=0x3000, rows=spad_rows),
            vpu_command(profile, kind="mul", src1_row=0, src2_row=0x3000,
                        dst_row=0x2000, rows=spad_rows), Fence(),
        ))
    else:
        commands.extend((
            vpu_command(profile, kind=operation, src1_row=0, src2_row=0x1000,
                        dst_row=0x2000, rows=spad_rows), Fence(),
        ))
    for start in range(0, spad_rows, 16):
        commands.append(_transfer_rect(3, "result", start * 16,
                                       0x2000 + start,
                                       rows=min(16, spad_rows - start)))
    commands.append(Fence())
    return emit_c(commands, transport="rocket_rocc",
                  buffers=("lhs", "rhs", "result"),
                  symbol=symbol, internal=True)


def _broadcast_index(result_shape: tuple[int, ...], rhs_shape: tuple[int, ...],
                     rhs_axes: tuple[int, ...]) -> str:
    if len(rhs_shape) > len(result_shape):
        raise ValueError("VPU elementwise RHS cannot exceed result rank")
    if (len(rhs_axes) != len(rhs_shape) or
            tuple(sorted(set(rhs_axes))) != rhs_axes or
            any(axis < 0 or axis >= len(result_shape) for axis in rhs_axes) or
            any(dim not in (1, result_shape[axis])
                for axis, dim in zip(rhs_axes, rhs_shape))):
        raise ValueError("VPU elementwise RHS shape does not broadcast to result")
    if rhs_shape == result_shape and rhs_axes == tuple(range(len(result_shape))):
        return "(base + index)"
    strides = []
    for axis in range(len(rhs_shape)):
        step = 1
        for dim in rhs_shape[axis + 1:]:
            step *= dim
        strides.append(step)
    terms = []
    for axis, dim in enumerate(rhs_shape):
        out_axis = rhs_axes[axis]
        if dim == 1:
            continue
        out_step = 1
        for size in result_shape[out_axis + 1:]:
            out_step *= size
        coordinate = (f"((base + index) % {result_shape[out_axis]}u)" if out_step == 1
                      else f"((base + index) / {out_step}u % {result_shape[out_axis]}u)")
        terms.append(f"({coordinate} * {strides[axis]}u)")
    return " + ".join(terms) if terms else "0u"


def emit_vpu_elementwise(profile: dict, *, result_shape: tuple[int, ...],
                         rhs_shape: tuple[int, ...], dtype: str,
                         operation: str, symbol: str,
                         rhs_axes: tuple[int, ...] | None = None,
                         lhs_shape: tuple[int, ...] | None = None,
                         lhs_axes: tuple[int, ...] | None = None,
                         lhs_dtype: str | None = None,
                         rhs_dtype: str | None = None,
                         preserve_f32_nan_refusal: bool = False) -> AddKernel:
    """Emit contiguous BF16 VPU add, sub, mul, or rcp+mul division."""
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu")):
        raise ValueError("VPU elementwise needs the selected DIM16 Rocket MX VPU")
    if (not 1 <= len(result_shape) <= 4 or
            any(type(dim) is not int or dim <= 0 for dim in (*result_shape, *rhs_shape,
                                                              *(lhs_shape or ()))) or
            dtype not in {"bf16", "f32"} or operation not in {"add", "sub", "mul", "div"} or
            not symbol.isidentifier() or not symbol.isascii()):
        raise ValueError("VPU elementwise needs a static contiguous tensor and known operation")
    lhs_dtype = lhs_dtype or dtype
    rhs_dtype = rhs_dtype or dtype
    if lhs_dtype not in {"bf16", "f32"} or rhs_dtype not in {"bf16", "f32"}:
        raise ValueError("VPU elementwise needs BF16 or f32 source tensors")
    if rhs_axes is None:
        rhs_axes = tuple(range(len(result_shape) - len(rhs_shape), len(result_shape)))
    if lhs_shape is None:
        lhs_shape = result_shape
    if lhs_axes is None:
        lhs_axes = tuple(range(len(result_shape) - len(lhs_shape), len(result_shape)))
    lhs_index = _broadcast_index(result_shape, lhs_shape, lhs_axes)
    rhs_index = _broadcast_index(result_shape, rhs_shape, rhs_axes)
    elements = 1
    for dim in result_shape:
        elements *= dim
    chunk = min(elements, 8192)
    tail = elements % chunk
    sizes = (chunk,) if not tail else (chunk, tail)
    issuers = "\n".join(_issue(profile, elements=count,
                                symbol=f"{symbol}_issue_{count}", operation=operation)
                        for count in sizes)
    dispatch = "\n".join(
        f"    case {count}: {symbol}_issue_{count}(a, b, c); break;"
        for count in sizes)
    scalar = "float" if dtype == "f32" else "uint16_t"
    lhs_scalar = "float" if lhs_dtype == "f32" else "uint16_t"
    rhs_scalar = "float" if rhs_dtype == "f32" else "uint16_t"
    lhs_read = ("uint32_t word = {symbol}_narrow(lhs[{lhs_index}]);"
                if lhs_dtype == "f32" else "uint32_t word = lhs[{lhs_index}];")
    rhs_read = ("{symbol}_narrow(rhs[{rhs_index}])" if rhs_dtype == "f32"
                else "rhs[{rhs_index}]")
    lhs_guard = ("if ((word & 0x7f80u) == 0x7f80u && (word & 0x7fu)) __builtin_trap();"
                 if preserve_f32_nan_refusal and lhs_dtype == "bf16" else "")
    rhs_guard = (f"if (({symbol}_b[index] & 0x7f80u) == 0x7f80u && "
                 f"({symbol}_b[index] & 0x7fu)) __builtin_trap();"
                 if preserve_f32_nan_refusal and rhs_dtype == "bf16" else "")
    out_write = ("{symbol}_widen({symbol}_c[index])" if dtype == "f32"
                 else f"{symbol}_c[index]")
    source = f'''#include <stdint.h>
{issuers}
static uint16_t {symbol}_a[{chunk // 8 * 8 + (8 if chunk % 8 else 0)}] __attribute__((aligned(64)));
static uint16_t {symbol}_b[{chunk // 8 * 8 + (8 if chunk % 8 else 0)}] __attribute__((aligned(64)));
static uint16_t {symbol}_c[{chunk // 8 * 8 + (8 if chunk % 8 else 0)}] __attribute__((aligned(64)));
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
void {symbol}(const {lhs_scalar} *lhs, const {rhs_scalar} *rhs, {scalar} *out) {{
  if (!lhs || !rhs || !out) __builtin_trap();
  for (uint32_t base = 0; base < {elements}; base += {chunk}) {{
    uint32_t count = {elements} - base;
    if (count > {chunk}) count = {chunk};
    for (uint32_t index = 0; index < (count + 7u) / 8u * 8u; ++index) {{
      if (index >= count) {{
        {symbol}_a[index] = {symbol}_b[index] = 0;
        continue;
      }}
      {lhs_read.format(symbol=symbol, lhs_index=lhs_index)}
      {lhs_guard}
      {symbol}_a[index] = (uint16_t)word;
      {symbol}_b[index] = {rhs_read.format(symbol=symbol, rhs_index=rhs_index)};
      {rhs_guard}
    }}
    const void *a = {symbol}_a, *b = {symbol}_b, *c = {symbol}_c;
    switch (count) {{
{dispatch}
      default: __builtin_trap();
    }}
    for (uint32_t index = 0; index < count; ++index)
      out[base + index] = {out_write.format(symbol=symbol)};
  }}
}}
'''
    return AddKernel(source, elements, chunk, dtype,
                     "none" if rhs_shape == result_shape else "shape_broadcast")


def emit_vpu_add(profile: dict, *, elements: int, row_width: int,
                 dtype: str, broadcast: str, symbol: str) -> AddKernel:
    """Emit a same-shape or last-dimension BF16 add wrapper."""
    if (type(elements) is not int or elements <= 0 or
            type(row_width) is not int or row_width <= 0 or
            elements % row_width or broadcast not in {"none", "last_dimension"}):
        raise ValueError("VPU add needs a static contiguous tensor and known broadcast")
    shape = (elements // row_width, row_width)
    rhs_shape = (row_width,) if broadcast == "last_dimension" else shape
    generated = emit_vpu_elementwise(profile, result_shape=shape,
                                     rhs_shape=rhs_shape, dtype=dtype,
                                     operation="add", symbol=symbol)
    return AddKernel(generated.source, elements, generated.elements_per_chunk,
                     dtype, broadcast)
