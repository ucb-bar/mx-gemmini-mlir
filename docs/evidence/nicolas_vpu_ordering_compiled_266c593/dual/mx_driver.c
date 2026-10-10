#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t out_x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t out_y[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint16_t ref_x[64][VPU_LANES];
static uint16_t ref_y[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *a, const void *a2, const void *b, const void *x,
              const void *out_x, const void *out_y, const void *output);

int main(void) {
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(100, 154, 0);  // P in Nicolas's source
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }
  x[0][0] = 0x0000; x[0][1] = 0x8000; x[0][2] = 0x7f80;
  x[0][3] = 0xff80; x[0][4] = 0x7fc1; x[0][5] = 0x0001;
  mx_issue(a, a2, b, x, out_x, out_y, output);
  gemmini_fence();

  vpu_ref_exec(VPU_ADDS, ref_x, a2, 0, 64, 1, 0, 0x4040);
  vpu_ref_exec(VPU_EXP, ref_y, x, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, ref, ref_x, b, 64, 1, 0, 0);
  int xy_bad = 0, z_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      xy_bad += (out_x[r][l] != ref_x[r][l]) + (out_y[r][l] != ref_y[r][l]);
      z_bad += output[r][l] != ref[r][l];
    }
  printf("compiled ordering dual: %d XY mismatches, %d Z mismatches\n",
         xy_bad, z_bad);
  return xy_bad != 0 || z_bad != 0;
}
