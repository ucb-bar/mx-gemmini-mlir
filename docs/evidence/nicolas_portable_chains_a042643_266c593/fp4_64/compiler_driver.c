// Chained back-to-back fp4 (e2m1) matmul on MxGemminiRocketConfig: C2 = (A1 @ B1) @ B2.
//
// MM1 (A1 @ B1) requantizes to fp4 and leaves C1 RESIDENT in the scratchpad in the BLOCK-TILED
// operand-A layout (LOOP_WS_REQUANT_TILED). MM2 reuses C1 in place as operand A (nibble HW-tiled)
// and reuses MM1's output block-scales as MM2's input A-scales. fp4 is direct 4-bit codes (no LUT),
// 2 m-rows packed per byte (low nibble = even row). Unified Spike (-DSPIKE_SIM) / RTL (-DMX_ROCKET)
// instruction stream, mirroring matmul_tiled_fp4_64x64_requant.c + matmul_tiled_fp8_64x64_chain.c.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp4_64x64_chain.h"
void mx_issue_source_order(const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *);


#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16
#define ADDR_LEN 32

// LOOP_WS rs2 bit10: deposit requant->spad output BLOCK-TILED (operand layout) for in-place reuse.
#define LOOP_WS_REQUANT_TILED (1u << 10)

// C8: automatic scale residency (always on for the chain). MM1's requant writes C1's output
// act-scales directly into the on-chip act-scale window (transposed [GN][M]) via
// gemmini_mxquant_config_mvout_resident (MX_SCALE_RESIDENT = mxquant config rs1 bit 63), so MM2
// reads its A-scales in place -- no DRAM buffer, no SW transpose/reload. Data + scales resident.
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)
#define MXQUANT_CFG(...) gemmini_mxquant_config_mvout_resident(__VA_ARGS__)  // automatic resident scale reuse

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#undef GEMMINI_BUSY_ADDR
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }
#endif

typedef uint8_t elem_t;   // 2 fp4 codes per byte

static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) { count += x & 1; x >>= 1; }
  return count;
}

// Read a BLOCK-TILED output resident in the scratchpad back to a flat [M][N]-byte DRAM buffer.
// Reads CONTIGUOUSLY (config_st=DIM) into a temp then de-tiles in SW -- a strided de-tile mvout
// (config_st=N) makes the writer DMA zero-fill gap cache lines on RTL at N>=128 and corrupts the
// output. For fp4 pass M=packed-rows (M/2), N=cols: tile (i,nt) at spad (i*tiles_N+nt)*DIM.
static void mvout_detile(uint8_t *dst, uint32_t spad_base, int M, int N) {
  int tiles_I = M / DIM, tiles_N = N / DIM;
  int total_rows = M * N / DIM;
  static uint8_t tiled[128 * 128];
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(tiled + r * DIM, spad_base + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(i * DIM + r) * N + nt * DIM + c] = tiled[((i * tiles_N + nt) * DIM + r) * DIM + c];
}

// Compare a resident nibble output (de-tiled to [M/2][N]) vs golden, nibble by nibble.
static int check_nibbles(const char *tag, uint8_t *got, const uint8_t *exp, int Mp, int N) {
  int errors = 0;
  for (int i = 0; i < Mp; i++)
    for (int j = 0; j < N; j++) {
      uint8_t g = got[i * N + j], e = exp[i * N + j];
      if ((g & 0xF) != (e & 0xF)) { errors++; if (errors <= 40) printf("%s[%d][%d].lo got %x exp %x\n", tag, i, j, g & 0xF, e & 0xF); }
      if ((g >> 4) != (e >> 4))   { errors++; if (errors <= 40) printf("%s[%d][%d].hi got %x exp %x\n", tag, i, j, g >> 4, e >> 4); }
    }
  return errors;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif

  // fp4 tiling: 32x32 mesh tiles -> tiles_I=M/32, tiles_J=N/32; K in 16-wide tiles.
  int tiles_I = MATMUL_M / DIM / 2;
  int tiles_J = MATMUL_N / DIM / 2;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  int SPAD_DEST1 = 2048;   // C1 resident here; fp4 64x64 C1 footprint = (M/2)*N/16 = 128 rows
  int SPAD_DEST2 = 4096;   // C2 here (clear of C1 and B2)

  static uint8_t C1_hw[MATMUL_M / 2][MATMUL_N];
  static uint8_t C2_hw[MATMUL_M / 2][MATMUL_N];
  uint32_t c1_scales[512] __attribute__((aligned(32))) = {0};
  uint32_t c2_scales[512] __attribute__((aligned(32))) = {0};
  memset(C1_hw, 0, sizeof(C1_hw));
  memset(C2_hw, 0, sizeof(C2_hw));

  static uint8_t c1_tiled_observed[MATMUL_M * MATMUL_N / 2] __attribute__((aligned(64)));
  static uint8_t c2_tiled[MATMUL_M * MATMUL_N / 2] __attribute__((aligned(64)));
  gemmini_flush(0);
  // The compiler issues both connected contractions and observes both C tiles.
  mx_issue_source_order(A_in_hw, A_scales_row, B_in, B_scales_col, B2_in,
           B2_scales_col, c1_scales, c1_tiled_observed, c2_scales, c2_tiled);
  for (int row = 0; row < MATMUL_M / 2; ++row)
    for (int col = 0; col < MATMUL_N; ++col) {
      int tile = (((row / 16) * (MATMUL_N / 16) + col / 16) * 16 + row % 16) * 16 + col % 16;
      C1_hw[row][col] = c1_tiled_observed[tile];
      C2_hw[row][col] = c2_tiled[tile];
    }
  int c1_err = check_nibbles("C1", (uint8_t *) C1_hw, (const uint8_t *) C1_out, MATMUL_M / 2, MATMUL_N);
  int c1_scale_err = 0;
  { uint8_t *sf = (uint8_t *) c1_scales;
    for (int i = 0; i < MATMUL_M; i++)
      for (int b = 0; b < MATMUL_GN; b++)
        if (sf[i * MATMUL_GN + b] != C1_scales_out[i][b]) {
          c1_scale_err++;
          if (c1_scale_err <= 40) printf("C1_scale[%d][%d] got %x exp %x\n", i, b, sf[i * MATMUL_GN + b], C1_scales_out[i][b]);
        } }
  if (c1_err == 0 && c1_scale_err == 0) printf("MM1 residency OK (C1 codes+scales exact).\n");
  else printf("MM1 residency FAILED: %d code, %d scale mismatches.\n", c1_err, c1_scale_err);

  // ================= MM2: C2 = requant(C1 @ B2), C1 read IN PLACE =================
  int c2_err = check_nibbles("C2", (uint8_t *) C2_hw, (const uint8_t *) C2_out, MATMUL_M / 2, MATMUL_N);
  int c2_scale_err = 0;
  { uint8_t *sf = (uint8_t *) c2_scales;
    for (int i = 0; i < MATMUL_M; i++)
      for (int b = 0; b < MATMUL_GN; b++)
        if (sf[i * MATMUL_GN + b] != C2_scales_out[i][b]) {
          c2_scale_err++;
          if (c2_scale_err <= 40) printf("C2_scale[%d][%d] got %x exp %x\n", i, b, sf[i * MATMUL_GN + b], C2_scales_out[i][b]);
        } }
  if (c2_err == 0 && c2_scale_err == 0) printf("MM2 chain OK (C2 codes+scales exact; C1 reused in place, scales reused).\n");
  else printf("MM2 chain FAILED: %d code, %d scale mismatches.\n", c2_err, c2_scale_err);

  int total = c1_err + c1_scale_err + c2_err + c2_scale_err;
  if (total == 0) printf("fp4 chain test PASSED (MM1 resident + reused as MM2 operand, scales reused).\n");
  else printf("fp4 chain test FAILED: %d total mismatches.\n", total);

#ifndef BAREMETAL
  exit(total != 0);
#else
  return total != 0;
#endif
}

void mx_issue(const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *);
void mx_issue_source_order(const void *a1_activation, const void *a1_scales, const void *b1_weight, const void *b1_scales, const void *b2_weight, const void *b2_scales, const void *c1_scales, const void *c1_tiled_observed, const void *c2_scales, const void *c2_tiled) {
  mx_issue(a1_activation, a1_scales, b1_scales, b1_weight, b2_scales, b2_weight, c1_scales, c1_tiled_observed, c2_scales, c2_tiled);
}
