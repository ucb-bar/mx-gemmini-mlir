"""Emit a bounded Rocket diagnostic from one source-bound MX GEMM contraction.

This emits a serial functional schedule for the checked-in FP8 four-wave
driver. It uses the source's A/B/E8M0/golden arrays and the target's verified
scratchpad and scale-buffer geometry. It does not lower arbitrary MLIR or
reproduce Muon's pipelined issue timing.
"""

from __future__ import annotations

from pathlib import Path

from .source_gemm import SourceGemm, plan_source_gemm
from .source_loop_trace import trace_bound_source_gemm_loops


def emit_source_fp8_baremetal(mlir_text: str, kernel: SourceGemm, *, profile: dict,
                             source_root: Path, rtl_root: Path) -> str:
    """Render a source-golden diagnostic for the selected 128x128x512 FP8 site."""
    trace = trace_bound_source_gemm_loops(
        mlir_text, kernel, profile=profile, source_root=source_root, rtl_root=rtl_root)
    plan = plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                            profile=profile)
    if (profile.get("transport") != "rocket_rocc" or profile["resources"]["lut"] or
            kernel.datatype != "FP8" or
            kernel.quant_output or kernel.acc_to_gmem or
            kernel.shape != (128, 128, 512) or kernel.tile != (128, 128, 128) or
            not kernel.data_header_present):
        raise ValueError("source bare-metal diagnostic supports the checked-in FP8 four-wave fullout driver")
    if (trace["wave_count"] != 4 or len(trace["scale_dma_packets"]) != 8 or
            plan["c_rows"] != 2048 or plan["a_rows"] != 1024 or plan["b_rows"] != 1024):
        raise ValueError("source GEMM layout differs from the bounded diagnostic")
    header = kernel.data_header.read_text()
    if "static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N]" not in header:
        raise ValueError("source FP8 header lacks its BF16 golden array")
    software_rows = trace["software_geometry"]["BANK_ROWS"]
    scratchpad_bytes = plan["scratchpad_bytes"]
    c_row = plan["c_spad_dest"]
    a_odd = plan["scratchpad_rows"] // 4
    b_odd = plan["scratchpad_rows"] - a_odd
    scale_half = trace["scale_memory"]["target_half_bytes"]
    return f'''// Source driver SHA-256: {trace["source_driver_sha256"]}
// MX target profile SHA-256: {trace["target_profile_sha256"]}
// Serial correctness diagnostic for the checked-in four-wave FP8 source kernel.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "{kernel.data_header.name}"

_Static_assert(DIM == 16 && BANK_NUM == 4 && BANK_ROWS == {software_rows},
               "selected Gemmini software geometry changed");
_Static_assert(BANK_NUM * BANK_ROWS * DIM == {scratchpad_bytes},
               "target scratchpad differs from the bound MX profile");
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t output_scales[512] __attribute__((aligned(64)));

int main(void) {{
  const uint32_t tiles_i = 8, tiles_j = 8, tiles_k = 8;
  const uint32_t c_row = {c_row};
  memset(C_hw, 0, sizeof C_hw);
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 0, 3, false);
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
  gemmini_fence();

  for (uint32_t wave = 0; wave < 4; ++wave) {{
    const uint32_t k_start = wave * 128;
    const uint32_t group = k_start / 32;
    const uint32_t odd = wave & 1;
    const uint32_t a_row = odd ? {a_odd} : 0;
    const uint32_t b_end = odd ? {b_odd} : {plan["scratchpad_rows"]};
    const uint32_t scale_dest = odd * {scale_half};

    // The source arrays are [K/32][M or N]; a 2-D DMA loads four rows.
    gemmini_mx_load_scales_2d(&A_scales_row[group][0], MATMUL_M, 4,
                              MATMUL_M, scale_dest, 0);
    gemmini_mx_load_scales_2d(&B_scales_col[group][0], MATMUL_N, 4,
                              MATMUL_N, scale_dest, 1);
    gemmini_fence();

    // Explicit mvin preserves the source's block-tiled operands while the
    // standalone Spike model does not execute loop-FSM DMA-only launches.
    gemmini_config_ld(MATMUL_K);
    for (uint32_t i = 0; i < tiles_i; ++i)
      for (uint32_t k = 0; k < tiles_k; ++k)
        gemmini_extended_mvin(&A_in[i * DIM][k_start + k * DIM],
                              a_row + (i * tiles_k + k) * DIM, DIM, DIM);
    gemmini_config_ld(MATMUL_N);
    const uint32_t b_start = b_end - tiles_k * tiles_j * DIM;
    for (uint32_t k = 0; k < tiles_k; ++k)
      for (uint32_t j = 0; j < tiles_j; ++j)
        gemmini_extended_mvin(&B_in[k_start + k * DIM][j * DIM],
                              b_start + (k * tiles_j + j) * DIM, DIM, DIM);
    gemmini_fence();

    gemmini_mxquant_config_mvout((uint64_t)output_scales,
                                  tiles_i, tiles_j, tiles_k, odd, odd, 1);
    gemmini_loop_ws_spad(tiles_i, tiles_j, tiles_k, 0, 0, 0,
                         a_row, b_end, 0, c_row, false, false, false, false,
                         wave != 0, NO_ACTIVATION, 0, 0, false,
                         wave == 3 ? 0x38 : 0xb8);
    gemmini_fence();
  }}

  // The final compute drains BF16 C into the target-derived scratchpad gap.
  gemmini_config_st(DIM);
  uint8_t *c_bytes = (uint8_t *)C_hw;
  for (uint32_t row = 0; row < {plan["c_rows"]}; row += DIM) {{
    gemmini_extended_mvout(c_bytes + row * DIM, c_row + row, DIM, DIM);
  }}
  gemmini_fence();

  int errors = 0;
  for (uint32_t i = 0; i < MATMUL_M; ++i)
    for (uint32_t j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {{
        if (errors < 8)
          printf("MISMATCH (%u,%u): got=0x%04x expected=0x%04x\\n",
                 i, j, C_hw[i][j], C_out_bf16[i][j]);
        ++errors;
      }}
  printf("source FP8 128x128x512: %d BF16 mismatches\\n", errors);
  return errors != 0;
}}
'''
