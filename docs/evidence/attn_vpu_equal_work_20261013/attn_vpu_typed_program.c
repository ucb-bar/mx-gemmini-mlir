#define ATTN_HOST_SOFTMAX 0
#define ATTN_ACC_ROWS 16
// One dense attention block on MxGemmini with the softmax on the VPU -- nothing but loads leaves the chip.
//
//   mvin   Q (op A, tiled), K^T (op B), V (op B)          scales: Q -> act half 0, K^T -> wgt half 0, V -> wgt half 1
//   mesh   S = Q @ K^T                  -> BF16, row-major in the scratchpad
//   VPU    S *= 1/sqrt(d); m = rowmax(S); S -= m; S = exp(S); l = rowsum(S); S *= 1/l     (P, in place)
//   SPAD_REQUANT  P -> E4M3 operand-A tiles, E8M0 scales resident in act half 0
//   mesh   O = P @ V                    -> BF16 -> mvout
//
// Every step is issued back to back; the reservation station's 4th (vector) queue orders the VPU and
// SPAD_REQUANT against the matmuls. Pass 1 also reads S, P and P's scales back mid-chain (the RS orders
// those too) and times the same softmax in fp32 on Rocket; pass 2 is the kernel alone, timed.
//   bit-exact: S vs S_GOLDEN (mesh model); S/P/O hashes vs Spike's (ATTN_EXPECT header, when present)
//   accuracy:  O vs O_REF_S (fp32 softmax over the mesh's own S) and O_REF_F (fp64 attention)
// -DATTN_STAGES fences every stage of pass 2 and prints per-stage cycles.
// Data: gen/gen_attn_vpu.py (default Sq 32, Sk 64, d 64; --tag _fa for Sq 64, Sk 256, d 128).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

#ifndef ATTN_HEADER
#define ATTN_HEADER "attn_vpu.h"
#define ATTN_EXPECT "attn_vpu_expect.h"   // Spike's hashes (recorded from a Spike run of this build)
#endif
#ifndef ATTN_ACC_ROWS
#define ATTN_ACC_ROWS ATTN_SQ  // rows sampled for the accuracy metrics (the exactness checks always cover all)
#endif
#ifndef ATTN_HOST_SOFTMAX
#define ATTN_HOST_SOFTMAX 1   // time + compare an fp32 softmax on Rocket (~8 cycles/element x 33: slow in RTL sim)
#endif
#include ATTN_HEADER

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs the VPU config (MX_ROCKET) or Spike\n"); return 0; }
#else

#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#define SQ ATTN_SQ
#define SK ATTN_SK
#define D  ATTN_D
#define OUT_BF16 3
#define SPAD_STORE 0x38

#define ROWS8(m, n)  ((m) * (n) / DIM)       // fp8 operand rows
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)   // BF16 rows
#define SP_Q      0
#define SP_KT     (SP_Q + ROWS8(SQ, D))
#define SP_KT_ARG (SP_KT + ROWS8(D, SK))
#define SP_S      SP_KT_ARG                  // S, then P (BF16) in place
#define SP_P      (SP_S + ROWS16(SQ, SK))    // P E4M3, operand-A tiles
#define SP_V      (SP_P + ROWS8(SQ, SK))
#define SP_V_ARG  (SP_V + ROWS8(SK, D))
#define SP_O      SP_V_ARG
#define SP_END    (SP_O + ROWS16(SQ, D))
#define SP_M      0x3000                     // row max / row sum: bank 3, apart from S
#define SP_L      (SP_M + SQ)
typedef char attn_spad_fits[(SP_END <= SP_M) ? 1 : -1];

static uint16_t S_hw[SQ][SK] __attribute__((aligned(64)));
static uint16_t P_hw[SQ][SK] __attribute__((aligned(64)));
static uint8_t  p_codes_hw[SQ * SK] __attribute__((aligned(64)));
static uint8_t  p_scales[SQ * SK / 32] __attribute__((aligned(64)));
static uint16_t O_hw[SQ][D] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(64)));
static float    p_host[SQ][SK];

void mx_issue_setup(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES, const void *Q_IN, const void *KT_IN, const void *V_IN, const void *scale_sink, const void *S_hw, const void *p_scales, const void *O_hw, const void *P_hw, const void *p_codes_hw);
void mx_issue_check(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES, const void *Q_IN, const void *KT_IN, const void *V_IN, const void *scale_sink, const void *S_hw, const void *p_scales, const void *O_hw, const void *P_hw, const void *p_codes_hw);
void mx_issue_timed(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES, const void *Q_IN, const void *KT_IN, const void *V_IN, const void *scale_sink, const void *S_hw, const void *p_scales, const void *O_hw, const void *P_hw, const void *p_codes_hw);

int *__errno(void) { static int e; return &e; }   // libm's errno under -nostdlib

static inline float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static uint64_t fnv(const void *p, size_t n) {   // FNV-1a over 64-bit words (n % 8 == 0)
  const uint64_t *w = (const uint64_t *)p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

static uint64_t st[8];
static int staged;
static void mark(int i) { if (staged) { gemmini_fence(); st[i] = read_cycles(); } }

// the kernel; check = 1 adds the mid-chain read-backs
static void attention(int check) {
  mark(0);
  if (check) mx_issue_check(Q_SCALES, KT_SCALES, V_SCALES, Q_IN, KT_IN, V_IN, scale_sink, S_hw, p_scales, O_hw, P_hw, p_codes_hw);
  else mx_issue_timed(Q_SCALES, KT_SCALES, V_SCALES, Q_IN, KT_IN, V_IN, scale_sink, S_hw, p_scales, O_hw, P_hw, p_codes_hw);
  gemmini_fence();
  st[6] = read_cycles();
}

static void rel_errs(double *eq, double *es, double *ef) {   // O vs the three references, one pass
  float num[3] = {0, 0, 0}, den[3] = {0, 0, 0};
  const uint32_t *refs[3] = {&O_REF_Q_F32[0][0], &O_REF_S_F32[0][0], &O_REF_F_F32[0][0]};
  for (int i = 0; i < ATTN_ACC_ROWS * D; i++) {
    const float o = bf(((const uint16_t *)O_hw)[i]);
    for (int k = 0; k < 3; k++) {
      float r; memcpy(&r, &refs[k][i], 4);
      num[k] += (o - r) * (o - r); den[k] += r * r;
    }
  }
  *eq = sqrtf(num[0] / den[0]); *es = sqrtf(num[1] / den[1]); *ef = sqrtf(num[2] / den[2]);
}

int main() {
  printf("attn_vpu: Sq=%d Sk=%d d=%d, softmax on the VPU (spad rows: S %d, P %d, O %d)\n", SQ, SK, D,
         ROWS16(SQ, SK), ROWS8(SQ, SK), ROWS16(SQ, D));
  gemmini_flush(0);
  mx_issue_setup(Q_SCALES, KT_SCALES, V_SCALES, Q_IN, KT_IN, V_IN, scale_sink, S_hw, p_scales, O_hw, P_hw, p_codes_hw);
  gemmini_fence();

  // ---- pass 1: correctness ----
  memset(O_hw, 0xa5, sizeof(O_hw)); memset(p_scales, 0xa5, sizeof(p_scales));
  staged = 0;
  attention(1);
  int s_diff = 0;   // 64-bit words (4 BF16) that differ
  for (int i = 0; i < SQ * SK / 4; i++) s_diff += ((const uint64_t *)S_hw)[i] != ((const uint64_t *)S_GOLDEN)[i];
  double worst_sum = 0;
  for (int m = 0; m < ATTN_ACC_ROWS; m++) {
    float s = 0;
    for (int n = 0; n < SK; n++) s += bf(P_hw[m][n]);
    if (fabsf(s - 1.0f) > worst_sum) worst_sum = fabsf(s - 1.0f);
  }
  uint64_t host_cyc = 0;
  double p_err = 0;
#if ATTN_HOST_SOFTMAX
  // the same softmax in fp32 on Rocket, from the same S, for time and for the VPU's softmax error
  const float scf = 1.0f / sqrtf((float)D);
  uint64_t h0 = read_cycles();
  for (int m = 0; m < SQ; m++) {
    float mx = -INFINITY, sum = 0;
    for (int n = 0; n < SK; n++) { p_host[m][n] = vpu_bf16_to_f(S_hw[m][n]) * scf; if (p_host[m][n] > mx) mx = p_host[m][n]; }
    for (int n = 0; n < SK; n++) { p_host[m][n] = expf(p_host[m][n] - mx); sum += p_host[m][n]; }
    for (int n = 0; n < SK; n++) p_host[m][n] /= sum;
  }
  host_cyc = read_cycles() - h0;
  for (int m = 0; m < SQ; m++)
    for (int n = 0; n < SK; n++) {
      double e = fabs((double)vpu_bf16_to_f(P_hw[m][n]) - p_host[m][n]);
      if (e > p_err) p_err = e;
    }
#endif
  const uint64_t hS = fnv(S_hw, sizeof(S_hw)), hP = fnv(P_hw, sizeof(P_hw)),
                 hPc = fnv(p_codes_hw, sizeof(p_codes_hw)) ^ (fnv(p_scales, sizeof(p_scales)) * 3), hO = fnv(O_hw, sizeof(O_hw));
  printf("check S vs mesh golden: %d/%d 4-element words differ\n", s_diff, SQ * SK / 4);
  printf("check hashes: S %016llx  P %016llx  Pq %016llx  O %016llx\n", (unsigned long long)hS,
         (unsigned long long)hP, (unsigned long long)hPc, (unsigned long long)hO);
  int fail = s_diff != 0;
#ifdef ATTN_EXPECT
#include ATTN_EXPECT
  const int hash_bad = (hS != EXP_HASH_S) + (hP != EXP_HASH_P) + (hPc != EXP_HASH_PQ) + (hO != EXP_HASH_O);
  printf("check hashes vs Spike: %s (%d of 4 differ)\n", hash_bad ? "MISMATCH" : "match", hash_bad);
  fail |= hash_bad != 0;
#else
  printf("check hashes vs Spike: no expectation header (first run on Spike records them)\n");
#endif
  double e_q, e_s, e_f;
  rel_errs(&e_q, &e_s, &e_f);
  printf("accuracy over %d of %d rows\n", ATTN_ACC_ROWS, SQ);
  printf("accuracy O: rel_fro %d ppm vs exact softmax + same MX steps | %d ppm vs fp32 softmax of the mesh S | "
         "%d ppm vs fp64 attention\n", (int)(e_q * 1e6), (int)(e_s * 1e6), (int)(e_f * 1e6));
  if (ATTN_HOST_SOFTMAX) printf("accuracy P: max |P_vpu - P_fp32| = %d ppm\n", (int)(p_err * 1e6));
  printf("accuracy P: max |rowsum - 1| = %d ppm\n", (int)(worst_sum * 1e6));

  // ---- pass 2: the kernel alone, timed ----
  gemmini_fence();
  st[0] = read_cycles();
#ifdef ATTN_STAGES
  staged = 1;
#endif
  attention(0);
  const uint64_t total = st[6] - st[0];
  const uint64_t ideal = 2ULL * SQ * SK * D / (DIM * DIM);
  printf("PERF attn_vpu: %llu cycles (mesh ideal %llu for QK^T + PV = %llu%%)\n",
         (unsigned long long)total, (unsigned long long)ideal, (unsigned long long)(ideal * 100 / total));
  if (ATTN_HOST_SOFTMAX) printf("PERF host fp32 softmax alone: %llu cycles\n", (unsigned long long)host_cyc);
#ifdef ATTN_STAGES
  printf("PERF stages: mvin %llu | QK^T %llu | VPU softmax (7 ops, %d rows) %llu | spad_requant %llu | PV %llu | mvout %llu\n",
         (unsigned long long)(st[1] - st[0]), (unsigned long long)(st[2] - st[1]), SQ * SK / 8,
         (unsigned long long)(st[3] - st[2]), (unsigned long long)(st[4] - st[3]),
         (unsigned long long)(st[5] - st[4]), (unsigned long long)(st[6] - st[5]));
#endif
  printf("attn_vpu %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
