#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t p[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *src1, const void *src2, const void *output);

int main(void) {
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      a[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(110, 140, 1);  // A2 in Nicolas's source
      b[r][l] = rand_bf16(110, 140, 1);
      p[r][l] = rand_bf16(100, 154, 0);
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }
  x[0][0] = 0x0000; x[0][1] = 0x8000; x[0][2] = 0x7f80;
  x[0][3] = 0xff80; x[0][4] = 0x7fc1; x[0][5] = 0x0001;
  p[0][0] = 0x0000; p[0][1] = 0x7f80; p[0][2] = 0x0001;
  mx_issue(a, 0, output);
  gemmini_fence();
  vpu_ref_exec(VPU_ADDS, ref, a, 0, 64,
               1, 0, 16448);
  int bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) bad++;
  printf("compiled VPU adds: %d mismatches\n", bad);
  return bad != 0;
}
