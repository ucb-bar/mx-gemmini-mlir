#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"
#include "mx_issue.h"
static uint16_t C_bf16[128][128] __attribute__((aligned(64)));
static uint8_t C_packed[64][128] __attribute__((aligned(64)));
static uint8_t C_scales[512] __attribute__((aligned(64)));
int main(void) {
  mx_issue(A_in_hw, A_lut, A_scales_row, C_bf16, C_lut, C_packed, C_scales, B_in, B_lut, B_scales_col);
  gemmini_fence();
  int bf16 = 0, packed = 0, scale = 0;
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      if (C_bf16[i][j] != C_out_bf16[i][j]) ++bf16;
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 128; ++j)
      if (C_packed[i][j] != C_proj_hw[i][j]) ++packed;
  for (int i = 0; i < 128; ++i)
    for (int g = 0; g < 4; ++g)
      if (C_scales[i * 4 + g] != C_scales_row[g][i]) ++scale;
  printf("compiled Nicolas generic FP6 data: %d bf16 / 16384, %d packed / 8192, %d scale / 512 mismatches\n",
         bf16, packed, scale);
  return bf16 || packed || scale;
}
