"""Target-owned dense f32 adapter for a checked MX encoded-pointer issuer.

The generated entry implements Merlin's three-pointer catalog ABI. Its static
workspace is deliberately non-reentrant: the selected bare-metal execution
lane calls one kernel at a time. A concurrent runtime needs an explicit
workspace ABI before it can use this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_issuer import RuntimeIssuer


@dataclass(frozen=True)
class DenseKernel:
    source: str
    symbol: str
    logical_shape_mnk: tuple[int, int, int]
    padded_shape_mnk: tuple[int, int, int]
    workspace_bytes: int


def _identifier(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] == "_") and all(
        char.isalnum() or char == "_" for char in value
    )


def emit_dense_f32_kernel(issuer: RuntimeIssuer, logical_shape_mnk: tuple[int, int, int], *,
                          symbol: str, issuer_symbol: str = "mx_issue_encoded") -> DenseKernel:
    """Wrap one serial FP4/FP8 BF16 issuer with live padding and MXQuant packing.

    The issuer must have been independently checked against its profile and
    source payload. This function validates its buffer extents again before
    generating a dense adapter; it does not qualify a new shape or datapath.
    """
    if not _identifier(symbol) or not _identifier(issuer_symbol) or symbol == issuer_symbol:
        raise ValueError("MX dense adapter needs distinct C entry and issuer symbols")
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
    padded_a = (lm, lk) != (m, k)
    padded_b = (lk, ln) != (k, n)
    declarations = []
    if padded_a:
        declarations.append(f"static float padded_a[{m} * {k}] __attribute__((aligned(64)));\n")
    if padded_b:
        declarations.append(f"static float padded_b[{k} * {n}] __attribute__((aligned(64)));\n")
    for name, size in expected.items():
        dtype = "uint16_t" if name == "output_bf16" else "uint8_t"
        count = size // 2 if name == "output_bf16" else size
        declarations.append(
            f"static {dtype} {name}[{count}] __attribute__((aligned(64)));\n"
        )
    a_prepare = (f"""  for (uint32_t i = 0; i < {m}u * {k}u; ++i) padded_a[i] = 0.0f;
  for (uint32_t row = 0; row < {lm}u; ++row)
    for (uint32_t col = 0; col < {lk}u; ++col)
      padded_a[(size_t)row * {k}u + col] = lhs[(size_t)row * {lk}u + col];
""" if padded_a else "")
    b_prepare = (f"""  for (uint32_t i = 0; i < {k}u * {n}u; ++i) padded_b[i] = 0.0f;
  for (uint32_t row = 0; row < {lk}u; ++row)
    for (uint32_t col = 0; col < {ln}u; ++col)
      padded_b[(size_t)row * {n}u + col] = rhs[(size_t)row * {ln}u + col];
""" if padded_b else "")
    a_source = "padded_a" if padded_a else "lhs"
    b_source = "padded_b" if padded_b else "rhs"
    copy_output = (f"""  for (uint32_t row = 0; row < {lm}u; ++row)
    for (uint32_t col = 0; col < {ln}u; ++col) {{
      union {{ uint32_t bits; float value; }} word;
      word.bits = (uint32_t)output_bf16[(size_t)row * {n}u + col] << 16;
      out[(size_t)row * {ln}u + col] = word.value;
    }}
""")
    source = f"""/* MX dense f32 adapter: {issuer.precision}, qualified {m}x{n}x{k} schedule. */
#include <stddef.h>
#include <stdint.h>
#include "runtime_quant.h"

void {issuer_symbol}(const void *, const void *, const void *, const void *,
                     const void *, const void *);
{''.join(declarations)}
void {symbol}(const float *lhs, const float *rhs, float *out) {{
  if (!lhs || !rhs || !out) __builtin_trap();
{a_prepare}{b_prepare}  if (mx_pack_activation_f32({a_source}, {m}u, {k}u, MX_RUNTIME_{issuer.precision},
                             activation, activation_scales) ||
      mx_pack_weight_f32({b_source}, {k}u, {n}u, MX_RUNTIME_{issuer.precision},
                         weight, weight_scales)) __builtin_trap();
  {issuer_symbol}(activation, activation_scales, weight, weight_scales,
                  output_bf16, scratch_output_scales);
{copy_output}}}
"""
    workspace = sum(expected.values()) + (m * k * 4 if padded_a else 0) + (
        k * n * 4 if padded_b else 0)
    return DenseKernel(source, symbol, logical_shape_mnk, issuer.shape_mnk, workspace)
