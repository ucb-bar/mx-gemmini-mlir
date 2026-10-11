#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/vpu_ref.h"
#define N 64
static uint16_t A[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t A2[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t B[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t P[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t X[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t observed[30][N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t expected[30][N][VPU_LANES];
static uint16_t scratch[N][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *a, const void *a2, const void *b, const void *p,
              const void *x, const void *out_0, const void *out_1, const void *out_2, const void *out_3, const void *out_4, const void *out_5, const void *out_6, const void *out_7, const void *out_8, const void *out_9, const void *out_10, const void *out_11, const void *out_12, const void *out_13, const void *out_14, const void *out_15, const void *out_16, const void *out_17, const void *out_18, const void *out_19, const void *out_20, const void *out_21, const void *out_22, const void *out_23, const void *out_24, const void *out_25, const void *out_26, const void *out_27, const void *out_28, const void *out_29);
int main(void) {
  for (int r = 0; r < N; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      A[r][l] = rand_bf16(110, 140, 1);
      A2[r][l] = rand_bf16(110, 140, 1);
      B[r][l] = rand_bf16(110, 140, 1);
      P[r][l] = rand_bf16(100, 154, 0);
      uint16_t x;
      do x = rand_bf16(100, 133, 1); while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
      X[r][l] = x;
    }
  X[0][0] = 0x0000; X[0][1] = 0x8000; X[0][2] = 0x7f80;
  X[0][3] = 0xff80; X[0][4] = 0x7fc1; X[0][5] = 0x0001;
  P[0][0] = 0x0000; P[0][1] = 0x7f80; P[0][2] = 0x0001;
  memset(observed, 0xa5, sizeof(observed));
  mx_issue(A, A2, B, P, X, observed[0], observed[1], observed[2], observed[3], observed[4], observed[5], observed[6], observed[7], observed[8], observed[9], observed[10], observed[11], observed[12], observed[13], observed[14], observed[15], observed[16], observed[17], observed[18], observed[19], observed[20], observed[21], observed[22], observed[23], observed[24], observed[25], observed[26], observed[27], observed[28], observed[29]);
  vpu_ref_exec(VPU_ADD, expected[0], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_SUB, expected[1], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, expected[2], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, expected[3], A, A2, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, expected[4], A, B, 64, 4, 1, 0);
  vpu_ref_exec(VPU_MAX, expected[5], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MAX, expected[6], A, A2, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MAX, expected[7], A, B, 64, 4, 1, 0);
  vpu_ref_exec(VPU_SUB, expected[8], A, A2, 64, 4, 1, 0);
  vpu_ref_exec(VPU_EXPSUB, expected[9], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_EXPSUB, expected[10], A, B, 64, 4, 1, 0);
  vpu_ref_exec(VPU_EXPSUB, expected[11], A, A2, 64, 4, 1, 0);
  vpu_ref_exec(VPU_EXPSUM, expected[12], A, B, 64, 4, 1, 12288);
  vpu_ref_exec(VPU_RSUM, expected[13], expected[12], 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_EXPSUM, expected[14], A, A2, 64, 4, 1, 12288);
  vpu_ref_exec(VPU_RSUM, expected[15], expected[14], 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_ADDS, expected[16], A, 0, 64, 1, 0, 16448);
  vpu_ref_exec(VPU_MULS, expected[17], A, 0, 64, 1, 0, 16128);
  vpu_ref_exec(VPU_EXP, expected[18], X, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_RCP, expected[19], A, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_RSQRT, expected[20], P, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_RMAX, expected[21], A, 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_RAMAX, expected[22], A, 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_RSUM, expected[23], A, 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_RSUM, expected[24], A, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_ADD, scratch, A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MULS, scratch, scratch, 0, 64, 1, 0, 0x3f00);
  vpu_ref_exec(VPU_RMAX, expected[25], scratch, 0, 64, 4, 0, 0);
  vpu_ref_exec(VPU_ADD, expected[26], A, B, 64, 1, 0, 0);
  vpu_ref_exec(VPU_ADDS, expected[27], A2, 0, 64, 1, 0, 0x4040);
  vpu_ref_exec(VPU_EXP, expected[28], X, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, expected[29], expected[27], B, 64, 1, 0, 0);
  int bad[30] = {0};
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[0] += observed[0][r][l] != expected[0][r][l];
  printf("add: %d mismatches\n", bad[0]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[1] += observed[1][r][l] != expected[1][r][l];
  printf("sub: %d mismatches\n", bad[1]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[2] += observed[2][r][l] != expected[2][r][l];
  printf("mul: %d mismatches\n", bad[2]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[3] += observed[3][r][l] != expected[3][r][l];
  printf("mul same-bank: %d mismatches\n", bad[3]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[4] += observed[4][r][l] != expected[4][r][l];
  printf("mul bcast: %d mismatches\n", bad[4]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[5] += observed[5][r][l] != expected[5][r][l];
  printf("max: %d mismatches\n", bad[5]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[6] += observed[6][r][l] != expected[6][r][l];
  printf("max same-bank: %d mismatches\n", bad[6]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[7] += observed[7][r][l] != expected[7][r][l];
  printf("max bcast: %d mismatches\n", bad[7]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[8] += observed[8][r][l] != expected[8][r][l];
  printf("sub bcast sb: %d mismatches\n", bad[8]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[9] += observed[9][r][l] != expected[9][r][l];
  printf("expsub: %d mismatches\n", bad[9]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[10] += observed[10][r][l] != expected[10][r][l];
  printf("expsub bcast: %d mismatches\n", bad[10]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[11] += observed[11][r][l] != expected[11][r][l];
  printf("expsub sb: %d mismatches\n", bad[11]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[12] += observed[12][r][l] != expected[12][r][l];
  printf("expsum bcast: %d mismatches\n", bad[12]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[13] += observed[13][r][l] != expected[13][r][l];
  printf("expsum sums: %d mismatches\n", bad[13]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[14] += observed[14][r][l] != expected[14][r][l];
  printf("expsum sb: %d mismatches\n", bad[14]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[15] += observed[15][r][l] != expected[15][r][l];
  printf("expsum sb sums: %d mismatches\n", bad[15]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[16] += observed[16][r][l] != expected[16][r][l];
  printf("adds: %d mismatches\n", bad[16]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[17] += observed[17][r][l] != expected[17][r][l];
  printf("muls: %d mismatches\n", bad[17]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[18] += observed[18][r][l] != expected[18][r][l];
  printf("exp: %d mismatches\n", bad[18]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[19] += observed[19][r][l] != expected[19][r][l];
  printf("rcp: %d mismatches\n", bad[19]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[20] += observed[20][r][l] != expected[20][r][l];
  printf("rsqrt: %d mismatches\n", bad[20]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[21] += observed[21][r][l] != expected[21][r][l];
  printf("rmax: %d mismatches\n", bad[21]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[22] += observed[22][r][l] != expected[22][r][l];
  printf("ramax: %d mismatches\n", bad[22]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[23] += observed[23][r][l] != expected[23][r][l];
  printf("rsum: %d mismatches\n", bad[23]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[24] += observed[24][r][l] != expected[24][r][l];
  printf("rsum rlen1: %d mismatches\n", bad[24]);
  for (int r = 0; r < 16; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[25] += observed[25][r][l] != expected[25][r][l];
  printf("chain: %d mismatches\n", bad[25]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[26] += observed[26][r][l] != expected[26][r][l];
  printf("war mvin: %d mismatches\n", bad[26]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[27] += observed[27][r][l] != expected[27][r][l];
  printf("dual X: %d mismatches\n", bad[27]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[28] += observed[28][r][l] != expected[28][r][l];
  printf("dual Y: %d mismatches\n", bad[28]);
  for (int r = 0; r < 64; ++r) for (int l = 0; l < VPU_LANES; ++l) bad[29] += observed[29][r][l] != expected[29][r][l];
  printf("dual Z=X*B: %d mismatches\n", bad[29]);
  int total = 0;
  for (int i = 0; i < 30; ++i) total += bad[i];
  printf("compiled vpu_ops: %d mismatches across 30 output snapshots\n", total);
  return total != 0;
}
