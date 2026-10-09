// Source driver SHA-256: 18bfc5d0af4687bd965f546508d945adcfbfee5d242e6269ec442889888e02ed
// Source header SHA-256: b57ef75fc634acd21750db45202bf12ced04569701f74e0607e0f06c974da93e
// MX target profile SHA-256: 704ec0fb715064fc64c672439c24286f137c74d37dbb7394e4e4b03317ad9737
// Serial correctness diagnostic for the checked-in FP6 source kernel.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "mxgemm.data.fp6.m128n128k2048.h"

_Static_assert(DIM == 16 && BANK_NUM == 4 && BANK_ROWS == 4096,
               "selected Gemmini software geometry changed");
_Static_assert(BANK_NUM * BANK_ROWS * DIM == 262144,
               "target scratchpad differs from the bound MX profile");
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t output_scales[512] __attribute__((aligned(64)));

int main(void) {
  const uint32_t tiles_i = 4, tiles_j = 4, tiles_k = 8;
  const uint32_t c_row = 512;
  memset(C_hw, 0, sizeof C_hw);
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 1, 1, 3, true);
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));

  // The source uploads 64 row/column-specific lines for each LUT bank once.
  gemmini_mx_load_lut(&B_lut[0][0], 64, 0);
  gemmini_mx_load_lut(&A_lut[0][0], 64, 1);
  gemmini_mx_load_lut(&C_lut[0][0], 64, 2);
  gemmini_fence();

  for (uint32_t wave = 0; wave < 16; ++wave) {
    const uint32_t k_start = wave * 128;
    const uint32_t group = k_start / 32;
    const uint32_t odd = wave & 1;
    const uint32_t a_row = odd ? 4096 : 0;
    const uint32_t b_end = odd ? 12288 : 16384;
    const uint32_t scale_dest = odd * 4096;
    gemmini_mx_load_scales_2d(&A_scales_row[group][0], MATMUL_M, 4,
                              MATMUL_M, scale_dest, 0);
    gemmini_mx_load_scales_2d(&B_scales_col[group][0], MATMUL_N, 4,
                              MATMUL_N, scale_dest, 1);
    gemmini_fence();

    // A and B contain packed 4-bit indices, decoded by the uploaded E3M2 LUTs.
    gemmini_config_ld(MATMUL_K);
    for (uint32_t i = 0; i < tiles_i; ++i)
      for (uint32_t k = 0; k < tiles_k; ++k)
        gemmini_extended_mvin(&A_in_hw[i * DIM][k_start + k * DIM],
                              a_row + (i * tiles_k + k) * DIM, DIM, DIM);
    gemmini_config_ld(MATMUL_N / 2);
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
                         wave == 15 ? 0x38 : 0xb8);
    gemmini_fence();
  }

  gemmini_config_st(DIM);
  uint8_t *c_bytes = (uint8_t *)C_hw;
  for (uint32_t row = 0; row < 2048; row += DIM)
    gemmini_extended_mvout(c_bytes + row * DIM, c_row + row, DIM, DIM);
  gemmini_fence();

  int errors = 0;
  for (uint32_t i = 0; i < MATMUL_M; ++i)
    for (uint32_t j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 8)
          printf("MISMATCH (%u,%u): got=0x%04x expected=0x%04x\n",
                 i, j, C_hw[i][j], C_out_bf16[i][j]);
        ++errors;
      }
  printf("source FP6 128x128x2048: %d BF16 mismatches\n", errors);
  return errors != 0;
}
