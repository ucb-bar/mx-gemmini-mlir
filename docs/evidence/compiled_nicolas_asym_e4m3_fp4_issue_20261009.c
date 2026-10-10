// Generated from model2MLIR site functional:matmul and explicit source recipe.
// Source driver SHA-256: 2405a800afdb6554721e9c405cd2f3c1842d97c0a173ad3f53ff1ea309d74eaf
// Source header SHA-256: 020d5825f08a525ae274f4d0803607aaae89b2f7e8262c9a1fe74de2f7c5488e
// MX profile SHA-256: fb0e0e5a6bb0bca2fa215ac58fe81e6d430421c0a55a1859c6500b5575122826
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_asym_e4m3_fp4.h"

_Static_assert(DIM == 16 && BANK_NUM == 4 && BANK_ROWS == 4096,
               "selected asymmetric MX software geometry changed");
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t output_scales[512] __attribute__((aligned(64)));

int main(void) {
  const int tiles_i = MATMUL_M / 32;
  const int tiles_j = MATMUL_N / 32;
  const int tiles_k = MATMUL_K / DIM;
  const uint32_t a_base = 0;
  const uint32_t b_end = BANK_NUM * BANK_ROWS / 2;
  const uint32_t b_base = b_end - tiles_k * tiles_j * DIM;
  const uint32_t c_base = 128;
  memset(C_hw, 0, sizeof C_hw);
  gemmini_flush(0);
  // E4M3 activation via 4-bit LUT index, direct FP4 weight, BF16 output.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 2, 3, true);
  gemmini_mx_load_lut_dt((uint64_t)B_lut, MATMUL_N / 2, 0, 8);
  gemmini_mx_load_lut_dt((uint64_t)A_lut, MATMUL_M / 2, 1, 8);
  gemmini_mx_load_lut_dt((uint64_t)C_lut, MATMUL_M / 2, 2, 8);
  gemmini_mx_load_scales((uint64_t)A_scales_row, sizeof A_scales_row, 0);
  gemmini_mx_load_scales((uint64_t)B_scales_col, sizeof B_scales_col, 1);
  gemmini_fence();
  gemmini_config_ld(MATMUL_K);
  for (int i = 0; i < tiles_i; ++i)
    for (int k = 0; k < tiles_k; ++k)
      gemmini_extended_mvin((void *)&A_in_hw[i * DIM][k * DIM],
                            a_base + (i * tiles_k + k) * DIM, DIM, DIM);
  gemmini_config_ld(MATMUL_N / 2);
  for (int k = 0; k < tiles_k; ++k)
    for (int j = 0; j < tiles_j; ++j)
      gemmini_extended_mvin((void *)&B_in[k * DIM][j * DIM],
                            b_base + (k * tiles_j + j) * DIM, DIM, DIM);
  gemmini_fence();
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t)output_scales,
                                tiles_i, tiles_j, tiles_k, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_i, tiles_j, tiles_k,
      0, 0, 0, a_base, b_end, 0, c_base,
      false, false, false, false, false, NO_ACTIVATION,
      0, 0, false, 0x38);
  gemmini_fence();
  gemmini_config_st(DIM);
  uint8_t *dst = (uint8_t *)C_hw;
  for (int row = 0; row < MATMUL_M * MATMUL_N * 2 / DIM; row += DIM)
    gemmini_extended_mvout(dst + row * DIM, c_base + row, DIM, DIM);
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 8)
          printf("MISMATCH (%d,%d): got=0x%04x expected=0x%04x\n",
                 i, j, C_hw[i][j], C_out_bf16[i][j]);
        ++errors;
      }
  printf("lowered asymmetric E4M3xFP4 64x64x64: %d BF16 mismatches\n", errors);
  return errors != 0;
}
