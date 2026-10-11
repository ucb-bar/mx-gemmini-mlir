#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x96x64.h"
#include "mx_issue.h"
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
int main(void) {
  mx_issue(A_in, A_scales_row, C_hw, scratch_output_scales, B_in, B_scales_col);
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 8) printf("mismatch %d,%d got %x want %x\n",
                               i,j,C_hw[i][j],C_out_bf16[i][j]);
        ++errors;
      }
  printf("compiled Nicolas FP8 64x96x64: %d mismatches / %d BF16 values\n",
         errors, MATMUL_M * MATMUL_N);
  return errors != 0;
}
