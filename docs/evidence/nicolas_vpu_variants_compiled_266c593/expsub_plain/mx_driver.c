#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t sums[16][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint16_t sum_ref[16][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *a, const void *a2, const void *b,
              const void *output, const void *sums);

int main(void) {
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(100, 154, 0);  // P in Nicolas's source
      uint16_t x;
      do x = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
    }
  mx_issue(a, a2, b, output, sums);
  gemmini_fence();
  vpu_ref_exec(VPU_EXPSUB, ref, a, b, 64,
               1, 0, 0);
  int out_bad = 0, sum_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) out_bad++;
  printf("compiled variant expsub_plain: %d output mismatches, %d sum mismatches\n",
         out_bad, sum_bad);
  return out_bad != 0 || sum_bad != 0;
}
