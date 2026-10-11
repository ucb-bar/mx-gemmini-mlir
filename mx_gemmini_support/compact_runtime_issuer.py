"""Compact serial Rocket issuer for large regular FP4/FP8 BF16 GEMMs.

The C loops express the same command order as ``_regular_schedule_steps``.
Trace mode replaces RoCC instructions with callbacks so tests can compare
every funct and operand against the source-qualified unrolled scheduler.
"""

from __future__ import annotations

from .planned_runtime_issuer import PlannedRuntimeIssuer
from .source_gemm import plan_mx_gemm
from .target_profile import profile_sha256


_TEMPLATE = r'''#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");
#ifdef MX_TRACE_COMMANDS
extern void mx_trace_command(uint32_t funct, uint64_t rs1, uint64_t rs2);
extern void mx_trace_fence(void);
#define MX_CMD(F, A, B) mx_trace_command((F), (uint64_t)(A), (uint64_t)(B))
#define MX_FENCE() mx_trace_fence()
#else
#define MX_CMD(F, A, B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" \
    : : "r"((uint64_t)(A)), "r"((uint64_t)(B)) : "memory")
#define MX_FENCE() __asm__ volatile ("fence" ::: "memory")
#endif

static uint64_t mx_masked(const void *base, uint64_t offset, uint64_t mask) {
  uint64_t pointer = (uint64_t)(uintptr_t)base;
  if (!base || offset > UINT64_MAX - pointer) __builtin_trap();
  uint64_t address = pointer + offset;
  if (address & ~mask) __builtin_trap();
  return address;
}

void @SYMBOL@(const void *activation, const void *activation_scales,
              const void *weight, const void *weight_scales,
              const void *output_bf16, const void *scratch_output_scales) {
  if (!activation || !activation_scales || !weight || !weight_scales ||
      !output_bf16 || !scratch_output_scales) __builtin_trap();
  const uint64_t dim = @DIM@u, m = @M@u, n = @N@u, k = @K@u;
  const uint64_t tm = @TM@u, tn = @TN@u, tk = @TK@u;
  const uint64_t ti = @TI@u, tj = @TJ@u, tki = @TKI@u;
  const uint64_t a_stride = k, b_stride = @B_STRIDE@u;
  const uint64_t rows = @ROWS@u, c_spad = @C_SPAD@u;
  const uint64_t scale_half = 4096u;
  const uint64_t mask40 = (UINT64_C(1) << 40) - 1u;
  const uint64_t mask33 = (UINT64_C(1) << 33) - 1u;
  MX_CMD(7, 0, 0);
  MX_CMD(0, UINT64_C(@CONFIG_EX@), UINT64_C(1) << 48);
  MX_CMD(0, (dim << 16) | (1u << 8) | 1u, a_stride);
  MX_CMD(0, (dim << 16) | (1u << 8) | (1u << 3) | 1u, b_stride);
  MX_CMD(0, 2, n * 2u);
  MX_FENCE();
  for (uint64_t m_start = 0; m_start < m; m_start += tm) {
    for (uint64_t n_start = 0; n_start < n; n_start += tn) {
      for (uint64_t wave = 0; wave < k / tk; ++wave) {
        const uint64_t group = wave * (tk / 32u);
        const uint64_t groups = tk / 32u;
        const uint64_t selector = wave & 1u;
        const uint64_t dest = selector * scale_half;
        MX_CMD(27, mx_masked(activation_scales, group * m + m_start, mask40) |
                   (m << 40), (groups << 46) | (dest << 33) | tm);
        MX_CMD(27, mx_masked(weight_scales, group * n + n_start, mask40) |
                   (n << 40), (groups << 46) | (dest << 33) |
                   (UINT64_C(1) << 32) | tn);
        MX_FENCE();
        const uint64_t a_row = selector ? rows / 4u : 0u;
        const uint64_t b_end = selector ? rows - rows / 4u : rows;
        const uint64_t b_start = b_end - tki * tj * dim;
        MX_CMD(0, (dim << 16) | (1u << 8) | 1u, a_stride);
        for (uint64_t i = 0; i < ti; ++i)
          for (uint64_t ki = 0; ki < tki; ++ki) {
            const uint64_t offset = (m_start / @VALUES_PER_BYTE@u + i * dim) *
                                    a_stride + wave * tk + ki * dim;
            const uint64_t row = a_row + (i * tki + ki) * dim;
            MX_CMD(2, (uint64_t)(uintptr_t)activation + offset,
                   (dim << 48) | (dim << 32) | row);
          }
        MX_CMD(0, (dim << 16) | (1u << 8) | 1u, b_stride);
        for (uint64_t ki = 0; ki < tki; ++ki)
          for (uint64_t j = 0; j < tj; ++j) {
            const uint64_t offset = (wave * tk + ki * dim) * b_stride +
                                    n_start / @VALUES_PER_BYTE@u + j * dim;
            const uint64_t row = b_start + (ki * tj + j) * dim;
            MX_CMD(2, (uint64_t)(uintptr_t)weight + offset,
                   (dim << 48) | (dim << 32) | row);
          }
        MX_FENCE();
        const uint64_t selector_bits = (selector << 60) | (selector << 61) |
            (tki << 51) | (tj << 42) | (ti << 33);
        MX_CMD(26, mx_masked(scratch_output_scales, 0, mask33) | selector_bits, 1);
        MX_CMD(9, 0, (tki << 32) | (tj << 16) | ti);
        MX_CMD(24, a_row, b_end);
        const uint64_t last = wave + 1u == k / tk;
        MX_CMD(8, wave != 0u, (c_spad << 32) | 0x200u |
               (last ? 0x38u : 0xb8u));
        MX_FENCE();
      }
      MX_CMD(0, 2, dim);
@READOUT@
      MX_FENCE();
    }
  }
}
'''


_ROW_MAJOR_READOUT = r'''      for (uint64_t local_row = 0; local_row < tm; ++local_row) {
        const uint64_t offset = ((m_start + local_row) * n + n_start) * 2u;
        const uint64_t row = c_spad + local_row * @ROWS_PER_LOGICAL@u;
        MX_CMD(3, (uint64_t)(uintptr_t)output_bf16 + offset,
               (UINT64_C(@ROWS_PER_LOGICAL@) << 48) | (dim << 32) | row);
      }'''


_SINGLE_TILE_READOUT = r'''      for (uint64_t row_offset = 0; row_offset < @C_ROWS@u;
           row_offset += dim) {
        MX_CMD(3, (uint64_t)(uintptr_t)output_bf16 + row_offset * dim,
               (dim << 48) | (dim << 32) | (c_spad + row_offset));
      }'''


def emit_compact_runtime_issuer(profile: dict, shape_mnk: tuple[int, int, int],
                                tile_mnk: tuple[int, int, int], *, precision: str,
                                symbol: str = "mx_issue_encoded") -> PlannedRuntimeIssuer:
    """Generate O(1)-source-size C loops for a checked serial matrix plan."""
    if precision not in {"FP4", "FP8"} or profile.get("transport") != "rocket_rocc":
        raise ValueError("compact MX issuer needs direct FP4/FP8 Rocket RoCC")
    if (len(shape_mnk) != 3 or len(tile_mnk) != 3 or
            any(type(value) is not int or value <= 0 for value in
                (*shape_mnk, *tile_mnk))):
        raise ValueError("compact MX issuer needs positive static shape and tile")
    if (not symbol or not (symbol[0].isalpha() or symbol[0] == "_") or
            any(not (char.isalnum() or char == "_") for char in symbol)):
        raise ValueError("compact MX issuer needs a safe C symbol")
    m, n, k = shape_mnk
    tm, tn, tk = tile_mnk
    dim = profile["geometry"]["mesh_columns"]
    values_per_byte = 2 if precision == "FP4" else 1
    pe = dim * values_per_byte
    if (m % tm or n % tn or k % tk or tk % dim or
            profile["resources"]["scale_block_elements"] != 32 or
            profile["resources"]["scale_mem_config"]["size_bytes"] != 16384 or
            max(tm, tn) * (tk // 32) > 4096):
        raise ValueError("compact MX shape or scale geometry is unsupported")
    plan = plan_mx_gemm(
        shape=shape_mnk, tile=tile_mnk, datatype=precision, quant_output=False,
        acc_to_gmem=False, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
        profile=profile)
    tiles = len(plan.get("output_tiles", ())) or 1
    if tiles > 1:
        if tn * 2 % dim or not 1 <= tn * 2 // dim <= 16:
            raise ValueError("compact MX row-major BF16 readout is unsupported")
        plan = {**plan, "bf16_output_layout": "row_major_bf16"}
    bits = 4 if precision == "FP4" else 8
    lengths = {
        "activation": m * k * bits // 8,
        "activation_scales": m * k // 32,
        "weight": k * n * bits // 8,
        "weight_scales": k * n // 32,
        "output_bf16": m * n * 2,
        "scratch_output_scales": 2048,
    }
    if (m * k % 32 or k * n % 32 or
            any(value <= 0 or value > 0x7fffffff for value in lengths.values())):
        raise ValueError("compact MX buffer extents overflow or lack scale groups")
    fmt = 2 if precision == "FP4" else 0
    config_ex = (1 << 16) | (fmt << 12) | (fmt << 10) | (1 << 2) | (3 << 14)
    readout = _ROW_MAJOR_READOUT if tiles > 1 else _SINGLE_TILE_READOUT
    source = _TEMPLATE.replace("@READOUT@", readout)
    substitutions = {
        "SYMBOL": symbol, "DIM": dim, "M": m, "N": n, "K": k,
        "TM": tm, "TN": tn, "TK": tk,
        "TI": tm // pe, "TJ": tn // pe, "TKI": tk // dim,
        "B_STRIDE": n if precision == "FP8" else n // 2,
        "ROWS": plan["scratchpad_rows"], "C_SPAD": plan["c_spad_dest"],
        "CONFIG_EX": config_ex, "VALUES_PER_BYTE": values_per_byte,
        "ROWS_PER_LOGICAL": tn * 2 // dim, "C_ROWS": plan["c_rows"],
    }
    for key, value in substitutions.items():
        source = source.replace(f"@{key}@", str(value))
    if "@" in source:
        raise ValueError("compact MX emitter left an unresolved template value")
    return PlannedRuntimeIssuer(source, shape_mnk, tile_mnk, precision,
                                profile_sha256(profile), lengths, plan)
