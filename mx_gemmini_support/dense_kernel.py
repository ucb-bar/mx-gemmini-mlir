"""Target-owned dense adapters for a checked MX encoded-pointer issuer.

The generated entry implements Merlin's three-pointer catalog ABI. Its static
workspace is deliberately non-reentrant: the selected bare-metal execution
lane calls one kernel at a time. A concurrent runtime needs an explicit
workspace ABI before it can use this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planned_runtime_issuer import PlannedRuntimeIssuer
from .runtime_issuer import RuntimeIssuer


@dataclass(frozen=True)
class DenseKernel:
    source: str
    symbol: str
    logical_shape_mnk: tuple[int, int, int]
    padded_shape_mnk: tuple[int, int, int]
    workspace_bytes: int
    persistent_bytes: int = 0


@dataclass(frozen=True)
class DenseBatchKernel:
    source: str
    symbol: str
    logical_shape_bmnk: tuple[int, int, int, int]
    padded_shape_mnk: tuple[int, int, int]
    workspace_bytes: int


def _identifier(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] == "_") and all(
        char.isalnum() or char == "_" for char in value
    )


def emit_shared_workspace_definition(size: int, *, symbol: str) -> str:
    """Define one aligned serial workspace shared by all catalog entries."""
    if not _identifier(symbol) or type(size) is not int or not 1 <= size <= 0x7fffffff:
        raise ValueError("MX shared workspace needs a safe symbol and bounded size")
    return (f"unsigned char {symbol}[{size}] "
            "__attribute__((aligned(64)));\n")


def emit_dense_f32_kernel(issuer: RuntimeIssuer | PlannedRuntimeIssuer,
                          logical_shape_mnk: tuple[int, int, int], *,
                          symbol: str, issuer_symbol: str = "mx_issue_encoded",
                          shared_workspace_symbol: str | None = None,
                          prepacked_weight_symbols: tuple[str, str] | None = None,
                          cache_immutable_weight: bool = False,
                          prepared_activation: bool = False,
                          retain_prepared_rhs: bool = False,
                          virtual_zero_padding: bool = False,
                          output_dtype: str = "f32") -> DenseKernel:
    """Wrap one serial FP4/FP8 BF16 issuer with live padding and MXQuant packing.

    The issuer must have been checked against its profile. A payload-bound
    issuer has independent source qualification; a shape-planned issuer still
    needs a separate numerical execution gate. Buffer extents are checked
    again before generating the adapter.

    BF16 output is an explicit private-edge transport contract. An exactly
    sized readout writes directly to the caller's destination; padded output
    is cropped without widening. Callers must qualify downstream rounding and
    exceptional behavior before selecting this alternative boundary.
    """
    from merlin.llvmlower.dense_padding import DenseZeroPadding, emit_dense_zero_padding

    if type(virtual_zero_padding) is not bool:
        raise ValueError("MX virtual zero padding needs an explicit boolean selection")
    if not _identifier(symbol) or not _identifier(issuer_symbol) or symbol == issuer_symbol:
        raise ValueError("MX dense adapter needs distinct C entry and issuer symbols")
    if shared_workspace_symbol is not None and not _identifier(shared_workspace_symbol):
        raise ValueError("MX dense adapter needs a safe shared workspace symbol")
    if prepacked_weight_symbols is not None and (
            len(prepacked_weight_symbols) != 2 or
            any(not _identifier(name) for name in prepacked_weight_symbols) or
            len(set(prepacked_weight_symbols)) != 2):
        raise ValueError("MX prepacked weight needs two distinct safe symbols")
    if cache_immutable_weight and prepacked_weight_symbols is not None:
        raise ValueError("MX fixed weight must be either cached or prepacked")
    if prepared_activation and prepacked_weight_symbols is None:
        raise ValueError("MX prepared activation entry requires immutable prepacked weights")
    if retain_prepared_rhs and not prepared_activation:
        raise ValueError("MX retained prepared RHS is a shared-activation diagnostic only")
    if output_dtype not in {"f32", "bf16"}:
        raise ValueError("MX dense readout requires f32 or bf16 output")
    if issuer.precision not in {"FP4", "FP8"}:
        raise ValueError("MX dense adapter requires a direct FP4 or FP8 issuer")
    if (len(logical_shape_mnk) != 3 or len(issuer.shape_mnk) != 3 or
            any(type(value) is not int or value <= 0 for value in
                (*logical_shape_mnk, *issuer.shape_mnk))):
        raise ValueError("MX dense adapter needs positive static matrix dimensions")
    lm, ln, lk = logical_shape_mnk
    m, n, k = issuer.shape_mnk
    if lm > m or ln > n or lk > k:
        raise ValueError("logical MX dimensions exceed the qualified physical schedule")
    bits = 4 if issuer.precision == "FP4" else 8
    expected = {
        "activation": m * k * bits // 8,
        "activation_scales": m * k // 32,
        "weight": k * n * bits // 8,
        "weight_scales": k * n // 32,
        "output_bf16": m * n * 2,
        "scratch_output_scales": 2048,
    }
    if any(value <= 0 for value in expected.values()) or dict(issuer.buffer_bytes) != expected:
        raise ValueError("MX issuer buffer extents disagree with dense tensor shapes")
    if any(value > 0x7fffffff for value in
           (m, n, k, lm, ln, lk, m * n, m * k, k * n)):
        raise ValueError("MX dense adapter exceeds the selected 32-bit runtime extent")
    zero_rows = virtual_zero_padding and lm < m
    a_padding = DenseZeroPadding((lm, lk), (m, k), (0,) if zero_rows else ())
    b_padding = DenseZeroPadding((lk, ln), (k, n))
    padded_a, padded_b = a_padding.needs_copy, b_padding.needs_copy
    direct_bf16 = output_dtype == "bf16" and (lm, ln) == (m, n)
    buffers = []
    if padded_a and not prepared_activation:
        buffers.append(("padded_a", "float", a_padding.elements * 4, a_padding.elements))
    if padded_b and prepacked_weight_symbols is None:
        buffers.append(("padded_b", "float", k * n * 4, k * n))
    for name, size in expected.items():
        if direct_bf16 and name == "output_bf16":
            continue
        if prepared_activation and name in {"activation", "activation_scales"}:
            continue
        if (prepacked_weight_symbols is not None or cache_immutable_weight) and name in {
                "weight", "weight_scales"}:
            continue
        dtype = "uint16_t" if name == "output_bf16" else "uint8_t"
        count = size // 2 if name == "output_bf16" else size
        buffers.append((name, dtype, size, count))
    local_buffers = ""
    if shared_workspace_symbol is None:
        declarations = [
            f"static {dtype} {name}[{count}] __attribute__((aligned(64)));\n"
            for name, dtype, _size, count in buffers
        ]
        workspace = sum(size for _name, _dtype, size, _count in buffers)
    else:
        declarations = [f"extern unsigned char {shared_workspace_symbol}[];\n"]
        offset = 0
        lines = []
        for name, dtype, size, _count in buffers:
            offset = (offset + 63) // 64 * 64
            lines.append(
                f"  {dtype} *{name} = ({dtype} *)({shared_workspace_symbol} + {offset});\n")
            offset += size
        workspace = (offset + 63) // 64 * 64
        local_buffers = "".join(lines)
    if workspace > 0x7fffffff:
        raise ValueError("MX dense adapter workspace exceeds the selected runtime extent")
    a_prepare = emit_dense_zero_padding(a_padding, source="lhs", destination="padded_a")
    b_prepare = (emit_dense_zero_padding(b_padding, source="rhs", destination="padded_b")
                 if prepacked_weight_symbols is None else "")
    a_source = "padded_a" if padded_a else "lhs"
    b_source = "padded_b" if padded_b else "rhs"
    activation_call = (f"mx_pack_activation_zero_rows_f32({a_source}, {lm}u, {m}u, {k}u"
                       if zero_rows else f"mx_pack_activation_f32({a_source}, {m}u, {k}u")
    if prepacked_weight_symbols is None:
        if cache_immutable_weight:
            weight_codes = f"{symbol}_weight_cache"
            weight_scale = f"{symbol}_weight_scale_cache"
            external_weight = (f"static uint8_t {weight_codes}[{expected['weight']}] "
                               "__attribute__((aligned(64)));\n"
                               f"static uint8_t {weight_scale}[{expected['weight_scales']}] "
                               "__attribute__((aligned(64)));\n"
                               f"static unsigned char {symbol}_weight_ready;\n")
            init = (b_prepare + f"""  if (mx_pack_weight_f32({b_source}, {k}u, {n}u,
                         MX_RUNTIME_{issuer.precision}, {weight_codes},
                         {weight_scale})) __builtin_trap();
  {symbol}_weight_ready = 1;
""")
            b_prepare = ""
            pack = (f"""  if ({activation_call}, MX_RUNTIME_{issuer.precision},
                             activation, activation_scales)) __builtin_trap();
  if (!{symbol}_weight_ready) {{
{init}  }}
""")
        else:
            pack = (f"""  if ({activation_call}, MX_RUNTIME_{issuer.precision},
                             activation, activation_scales) ||
      mx_pack_weight_f32({b_source}, {k}u, {n}u, MX_RUNTIME_{issuer.precision},
                         weight, weight_scales)) __builtin_trap();
""")
            weight_codes, weight_scale = "weight", "weight_scales"
            external_weight = ""
    else:
        weight_codes, weight_scale = prepacked_weight_symbols
        external_weight = (f"extern const uint8_t {weight_codes}[];\n"
                           f"extern const uint8_t {weight_scale}[];\n")
        pack = (f"""  if ({activation_call}, MX_RUNTIME_{issuer.precision},
                             activation, activation_scales)) __builtin_trap();
""")
    if prepared_activation:
        a_prepare = ""
        pack = (f"  const uint8_t *activation = lhs;\n"
                f"  const uint8_t *activation_scales = lhs + {expected['activation']}u;\n")
    out_scalar = "uint16_t" if output_dtype == "bf16" else "float"
    parameters = (f"const uint8_t *lhs, const float *rhs, {out_scalar} *out"
                  if retain_prepared_rhs else
                  f"const uint8_t *lhs, {out_scalar} *out" if prepared_activation else
                  f"const float *lhs, const float *rhs, {out_scalar} *out")
    pointer_check = ("!lhs || !out" if prepared_activation and not retain_prepared_rhs
                     else "!lhs || !rhs || !out")
    copy_output = (f"""  for (uint32_t row = 0; row < {lm}u; ++row)
    for (uint32_t col = 0; col < {ln}u; ++col) {{
      union {{ uint32_t bits; float value; }} word;
      word.bits = (uint32_t)output_bf16[(size_t)row * {n}u + col] << 16;
      out[(size_t)row * {ln}u + col] = word.value;
    }}
""")
    if output_dtype == "bf16":
        copy_output = ("" if direct_bf16 else f"""  for (uint32_t row = 0; row < {lm}u; ++row)
    for (uint32_t col = 0; col < {ln}u; ++col)
      out[(size_t)row * {ln}u + col] = output_bf16[(size_t)row * {n}u + col];
""")
    source = f"""/* MX dense f32 adapter: {issuer.precision}, qualified {m}x{n}x{k} schedule. */
#include <stddef.h>
#include <stdint.h>
#include "runtime_quant.h"

void {issuer_symbol}(const void *, const void *, const void *, const void *,
                     const void *, const void *);
{external_weight}{''.join(declarations)}
void {symbol}({parameters}) {{
  if ({pointer_check}) __builtin_trap();
{local_buffers}{a_prepare}{b_prepare}{pack}  {issuer_symbol}(activation, activation_scales, {weight_codes}, {weight_scale},
                  {"out" if direct_bf16 else "output_bf16"}, scratch_output_scales);
{copy_output}}}
"""
    persistent = (expected["weight"] + expected["weight_scales"] + 1
                  if cache_immutable_weight else 0)
    return DenseKernel(source, symbol, logical_shape_mnk, issuer.shape_mnk,
                       workspace, persistent)


def emit_activation_preparation(issuer: RuntimeIssuer | PlannedRuntimeIssuer,
                                logical_shape_mnk: tuple[int, int, int], *,
                                symbol: str, virtual_zero_padding: bool = False) -> tuple[str, int, int]:
    """Emit deterministic packing into a caller-owned contiguous byte tensor.

    Codes precede scales with the exact extents consumed by the checked issuer.
    Padding scratch is private to this synchronous serial entry; packed output
    survives independently of every matrix adapter's shared workspace.
    """
    # Reuse the adapter's full shape/format/extent validation.
    emit_dense_f32_kernel(issuer, logical_shape_mnk, symbol=symbol + "_check",
                          virtual_zero_padding=virtual_zero_padding)
    from merlin.llvmlower.dense_padding import DenseZeroPadding, emit_dense_zero_padding
    if not _identifier(symbol):
        raise ValueError("MX activation preparation needs a safe symbol")
    lm, _, lk = logical_shape_mnk
    m, _, k = issuer.shape_mnk
    codes = issuer.buffer_bytes["activation"]
    size = codes + issuer.buffer_bytes["activation_scales"]
    zero_rows = virtual_zero_padding and lm < m
    plan = DenseZeroPadding((lm, lk), (m, k), (0,) if zero_rows else ())
    padded = plan.needs_copy
    declaration = (f"static float padded[{plan.elements}] __attribute__((aligned(64)));\n"
                   if padded else "")
    padding = emit_dense_zero_padding(plan, source="lhs", destination="padded")
    a_source = "padded" if padded else "lhs"
    call = (f"mx_pack_activation_zero_rows_f32({a_source}, {lm}u, {m}u, {k}u"
            if zero_rows else f"mx_pack_activation_f32({a_source}, {m}u, {k}u")
    source = f'''#include <stddef.h>
#include <stdint.h>
#include "runtime_quant.h"
{declaration}
void {symbol}(const float *lhs, uint8_t *out) {{
  if (!lhs || !out) __builtin_trap();
{padding}  if ({call},
      MX_RUNTIME_{issuer.precision}, out, out+{codes}u)) __builtin_trap();
}}
'''
    return source, size, plan.elements * 4


def emit_dense_f32_batch_kernel(issuer: RuntimeIssuer | PlannedRuntimeIssuer,
                                logical_shape_bmnk: tuple[int, int, int, int], *,
                                symbol: str,
                                issuer_symbol: str = "mx_issue_encoded",
                                shared_workspace_symbol: str | None = None,
                                virtual_zero_padding: bool = False) -> DenseBatchKernel:
    """Call one checked matrix kernel for each disjoint dense batch slice."""
    if (len(logical_shape_bmnk) != 4 or
            any(type(value) is not int or value <= 0 for value in logical_shape_bmnk)):
        raise ValueError("MX batch adapter needs positive B,M,N,K")
    batch, m, n, k = logical_shape_bmnk
    if any(value > 0x7fffffff for value in (batch * m * k, batch * k * n,
                                            batch * m * n)):
        raise ValueError("MX batch adapter exceeds the selected runtime extent")
    slice_symbol = f"{symbol}_slice"
    slice_kernel = emit_dense_f32_kernel(
        issuer, (m, n, k), symbol=slice_symbol, issuer_symbol=issuer_symbol,
        shared_workspace_symbol=shared_workspace_symbol, virtual_zero_padding=virtual_zero_padding)
    source = slice_kernel.source + f"""
/* Whole-batch Merlin ABI: each slice has independent dense row-major storage. */
void {symbol}(const float *lhs, const float *rhs, float *out) {{
  if (!lhs || !rhs || !out) __builtin_trap();
  for (size_t batch = 0; batch < {batch}u; ++batch)
    {slice_symbol}(lhs + batch * {m}u * {k}u,
                   rhs + batch * {k}u * {n}u,
                   out + batch * {m}u * {n}u);
}}
"""
    return DenseBatchKernel(source, symbol, logical_shape_bmnk,
                            issuer.shape_mnk, slice_kernel.workspace_bytes)
