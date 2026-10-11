#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp4_128x128x64_dim32.h"
#include "mx_issue.h"
static uint8_t C_hw[MATMUL_M/2][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
int main(void) {
  mx_issue(A_in_hw, A_scales_row, C_hw, scratch_output_scales, B_in, B_scales_col);
  gemmini_fence();
  int code_errors = 0, scale_errors = 0;
  for (int i = 0; i < MATMUL_M/2; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out[i][j]) {
        if (code_errors < 8) printf("packed-byte mismatch %d,%d got %x want %x\n",
                                    i,j,C_hw[i][j],C_out[i][j]);
        ++code_errors;
      }
  for (int i = 0; i < MATMUL_M; ++i)
    for (int g = 0; g < MATMUL_GN; ++g)
      if (scratch_output_scales[i*MATMUL_GN+g] != C_scales_out[i][g]) {
        if (scale_errors < 8) printf("scale mismatch %d,%d got %x want %x\n",
                                     i,g,scratch_output_scales[i*MATMUL_GN+g],
                                     C_scales_out[i][g]);
        ++scale_errors;
      }
  printf("compiled Nicolas FP4 128x128x64 requant: %d packed-byte mismatches / %d, %d scale mismatches / %d\n",
         code_errors, MATMUL_M/2*MATMUL_N, scale_errors, MATMUL_M*MATMUL_GN);
  return code_errors != 0 || scale_errors != 0;
}
