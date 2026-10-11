#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t p[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t middle[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t result[16][VPU_LANES] __attribute__((aligned(64)));
static uint16_t middle_ref[64][VPU_LANES];
static uint16_t result_ref[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *a, const void *b, const void *middle, const void *result);

int main(void) {
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      p[r][l] = rand_bf16(100, 154, 0);
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }
  mx_issue(a, b, middle, result);
  gemmini_fence();
  vpu_ref_exec(VPU_ADD, middle_ref, a, b, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MULS, middle_ref, middle_ref, 0, 64, 1, 0, 0x3f00);
  vpu_ref_exec(VPU_RMAX, result_ref, middle_ref, 0, 64, 4, 0, 0);
  int middle_bad = 0, result_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      middle_bad += middle[r][l] != middle_ref[r][l];
  for (int r = 0; r < 16; r++)
    for (int l = 0; l < VPU_LANES; l++)
      result_bad += result[r][l] != result_ref[r][l];
  printf("compiled VPU chain: %d middle mismatches, %d result mismatches\n",
         middle_bad, result_bad);
  return middle_bad || result_bad;
}
