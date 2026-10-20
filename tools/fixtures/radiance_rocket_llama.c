/* Rocket port of radiance-kernels nightly/llama_e2e's unfused phase program.
 *
 * The descriptor order, MX layouts, FP32 row arithmetic and BF16 boundaries
 * follow that source. Native DRAM loops replace Muon scratchpad commands;
 * VPU EXP replaces Muon fexp.h. FP32 arithmetic remains on Rocket because
 * this target's VPU stores BF16. This is a port, not Muon cycle emulation.
 */
#include "rocket_data.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define CMD(F, A, B)                                                           \
  __asm__ volatile(".insn r 0x7b, 3, " #F ", x0, %0, %1" ::"r"((uint64_t)(A)), \
                   "r"((uint64_t)(B))                                          \
                   : "memory")
static inline void fence(void) { __asm__ volatile("fence" ::: "memory"); }
static inline uint64_t cycle(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}
extern unsigned char rk_region1[], rk_region2[], rk_region3[];
extern unsigned char rk_guard1a[], rk_guard1b[], rk_guard2a[], rk_guard2b[],
    rk_guard3a[], rk_guard3b[];
#ifdef RK_EMBEDDING_LOOKUP
extern const float rk_embeddings[];
extern const int64_t rk_token_ids[];
static float rk_logits_f32[LL_M * LL_V] __attribute__((aligned(64)));
#endif
static void *ptr(uint32_t address) {
  if (address >= 0x20000000u && address - 0x20000000u < RK_REGION1_BYTES)
    return rk_region1 + (address - 0x20000000u);
  if (address >= 0x40000000u && address - 0x40000000u < RK_REGION2_BYTES)
    return rk_region2 + (address - 0x40000000u);
  if (address >= 0x60000000u && address - 0x60000000u < RK_REGION3_BYTES)
    return rk_region3 + (address - 0x60000000u);
  __builtin_trap();
}
static uint32_t bits(float f) {
  union {
    float f;
    uint32_t u;
  } x = {.f = f};
  return x.u;
}
static float value(uint32_t u) {
  union {
    uint32_t u;
    float f;
  } x = {.u = u};
  return x.f;
}
static uint16_t bf16(float f) {
  uint32_t u = bits(f);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
static float bf(uint16_t u) { return value((uint32_t)u << 16); }
static uint8_t fp8(float f) {
  uint32_t u = bits(f), s = (u >> 24) & 128u, ax = u & 0x7fffffffu;
  uint32_t normal = ((ax + 0x7ffffu + ((ax >> 20) & 1u)) >> 20) - (120u << 3);
  uint32_t sub = bits(value(ax) * 512.0f + 8388608.0f) - 0x4b000000u;
  return s | (ax < 0x3c800000u ? sub : normal);
}
static int block_exp(float amax) {
  int e = (int)((bits(amax) >> 23) & 255u) - 127;
  return e < -126 ? -126 : e;
}
static uint32_t a_code(uint32_t tm, uint32_t tk, uint32_t ks, uint32_t r,
                       uint32_t k) {
  return (((r / tm) * ks + k / tk) * tm + r % tm) * tk + k % tk;
}
static uint32_t a_scale(uint32_t tm, uint32_t tk, uint32_t ks, uint32_t r,
                        uint32_t k) {
  return (((r / tm) * ks + k / tk) * (tk / 32) + (k % tk) / 32) * tm + r % tm;
}
static void quant_group(const float *x, uint8_t *codes, uint32_t stride,
                        uint8_t *scale) {
#ifdef RK_ROUND_QUANT_INPUTS
  /* Match the published compiler's BF16 input boundary and minimum scale.
   * The source-default smoke model exercises the original FP32 encoder.
   */
  float rounded[32];
  for (unsigned j = 0; j < 32; j++)
    rounded[j] = bf(bf16(x[j]));
  x = rounded;
#endif
  float maximum = 0;
  for (unsigned j = 0; j < 32; j++)
    maximum = __builtin_fmaxf(maximum, __builtin_fabsf(x[j]));
  int e = block_exp(maximum);
#ifdef RK_ROUND_QUANT_INPUTS
  if (e < -23)
    e = -23;
#endif
  float multiplier = value((uint32_t)(127 - e) << 23);
  for (unsigned j = 0; j < 32; j++)
    codes[j * stride] = fp8(x[j] * multiplier);
  *scale = (uint8_t)(e + 127);
}
static void quant_a(const float *x, uint8_t *a, uint8_t *asc, uint32_t tm,
                    uint32_t tk, uint32_t k, uint32_t r, uint32_t g) {
  quant_group(x, a + a_code(tm, tk, k / tk, r, g * 32), 1,
              asc + a_scale(tm, tk, k / tk, r, g * 32));
}
#ifdef RK_QUANT_TEST
void rk_quant_test(const float *input, uint8_t *codes, uint8_t *scale) {
  quant_group(input, codes, 1, scale);
}
#else
static uint32_t native_loops;

/* Native Rocket loops read the source's pre-tiled A/B blocks directly.
 * The source's M/N/K tile partition is retained; each completed tile goes
 * straight to its descriptor's BF16 output with its original row stride.
 */
static void gemm(const uint32_t *d) {
  uint32_t mt_count = d[15], nt_count = d[16], ks = d[17], tm = d[18],
           tk = d[19];
  unsigned char *a = ptr(d[2]), *asc = ptr(d[3]), *b = ptr(d[6]),
                *bsc = ptr(d[7]), *c = ptr(d[10]);
  CMD(7, 0, 0);
  CMD(30, 0, 0);
  CMD(0, (1ull << 16) | (3ull << 14) | 4, 1ull << 48);
  CMD(0, (16ull << 16) | (1ull << 8) | 1, tk);
  CMD(0, (16ull << 16) | (1ull << 8) | (1ull << 3) | 1, 64);
  CMD(0, 2, d[12]);
  for (uint32_t bt = 0; bt < d[14]; bt++)
    for (uint32_t mt = 0; mt < mt_count; mt++)
      for (uint32_t nt = 0; nt < nt_count; nt++)
        for (uint32_t k = 0; k < ks; k++) {
          CMD(31, asc + bt * d[5] + (mt * ks + k) * tm * tk / 32,
              bsc + bt * d[9] + (nt * ks + k) * tk * 64 / 32);
          CMD(32, tm, 64);
          CMD(9, 0, ((uint64_t)(tk / 16) << 32) | (4ull << 16) | (tm / 16));
          CMD(10, a + bt * d[4] + (mt * ks + k) * tm * tk,
              b + bt * d[8] + (k * nt_count + nt) * tk * 64);
          CMD(11, 0,
              k == ks - 1 ? (uintptr_t)(c + bt * d[11] + mt * d[13] + nt * 128)
                          : 0);
          CMD(12, tk, 64);
          CMD(13, 0, d[12] / 2);
          uint32_t slot = 1 + (native_loops++ & 1u);
          CMD(8, (1ull << 18) | ((uint64_t)slot << 16) | (k > 0), 0);
        }
  fence();
}

/* BF16 input/output, up to 8192 elements; complete the VPU call before
 * reusing its scratchpad as a matrix operand or output.
 */
static void exp_bf16(const uint16_t *input, uint16_t *output, uint32_t n) {
  CMD(7, 0, 0);
  CMD(0, (16ull << 16) | (1ull << 8) | 1, 16);
  CMD(0, 2, 16);
  for (uint32_t base = 0; base < n; base += 8192) {
    uint32_t count = n - base;
    if (count > 8192)
      count = 8192;
    uint32_t rows = count / 8;
    for (uint32_t r = 0; r < rows; r += 16) {
      uint32_t nr = rows - r;
      if (nr > 16)
        nr = 16;
      CMD(2, input + base + r * 8, ((uint64_t)nr << 48) | (16ull << 32) | r);
    }
    CMD(33, (0x2000ull << 28) | ((uint64_t)rows << 42), 5 | (1ull << 5));
    fence();
    for (uint32_t r = 0; r < rows; r += 16) {
      uint32_t nr = rows - r;
      if (nr > 16)
        nr = 16;
      CMD(3, output + base + r * 8,
          ((uint64_t)nr << 48) | (16ull << 32) | (0x2000 + r));
    }
    fence();
  }
}
static uint16_t exp_input[16384] __attribute__((aligned(64)));
static uint16_t exp_output[16384] __attribute__((aligned(64)));
static float row[16384] __attribute__((aligned(64)));

static void prenorm(const uint32_t *d) {
  uint16_t *x = ptr(d[2]), *add = ptr(d[3]), *h = ptr(d[4]), *gamma = ptr(d[5]);
  uint8_t *a = ptr(d[6]), *asc = ptr(d[7]);
  uint32_t D = d[8], M = d[9], tm = d[10], tk = d[11];
  for (uint32_t r = 0; r < M; r++) {
    float sums[16] = {0};
    /* Same 16 lane partial sums and final lane order as prenorm_rows. */
    for (uint32_t lane = 0; lane < 16; lane++)
      for (uint32_t w0 = lane; w0 < D / 2; w0 += 128)
        for (uint32_t j = 0; j < 8; j++) {
          uint32_t col = 2 * (w0 + j * 16);
          float p = bf(bf16(bf(x[r * D + col]) + bf(add[r * D + col])));
          float q = bf(bf16(bf(x[r * D + col + 1]) + bf(add[r * D + col + 1])));
          h[r * D + col] = bf16(p);
          h[r * D + col + 1] = bf16(q);
          sums[lane] += p * p + q * q;
        }
    float ss = 0;
    for (unsigned lane = 0; lane < 16; lane++)
      ss += sums[lane];
    float inv = 1.0f / __builtin_sqrtf(ss / (float)D + value(d[12]));
    for (uint32_t c = 0; c < D; c++)
      row[c] = (bf(h[r * D + c]) * inv) * bf(gamma[c]);
    for (uint32_t g = 0; g < D / 32; g++)
      quant_a(row + g * 32, a, asc, tm, tk, D, r, g);
  }
}
static void ewq(const uint32_t *d, int silu) {
  uint16_t *x = ptr(d[2]);
  uint8_t *a = ptr(d[7]), *asc = ptr(d[8]);
  uint32_t rs = d[3], K = d[5], M = d[6], tm = d[9], tk = d[10];
  for (uint32_t r = 0; r < M; r++) {
    if (silu) {
      for (uint32_t c = 0; c < K; c++)
        exp_input[c] = x[r * rs + c] ^ 0x8000u;
      exp_bf16(exp_input, exp_output, K);
    }
    for (uint32_t c = 0; c < K; c++) {
      float v = bf(x[r * rs + c]);
      row[c] =
          silu ? (v / (1.0f + bf(exp_output[c]))) * bf(x[r * rs + K + c]) : v;
    }
    for (uint32_t g = 0; g < K / 32; g++)
      quant_a(row + g * 32, a, asc, tm, tk, K, r, g);
  }
}
static void rope(const uint32_t *d) {
  uint16_t *qkv = ptr(d[2]);
  uint32_t nqkv = d[3], hq = d[4], hkv = d[5], hd = d[6], half = hd / 2;
  float *cos = ptr(d[7]), *sin = ptr(d[8]);
  uint32_t M = d[22], SK = d[23], tm = d[24], gq = hq / hkv;
  uint8_t *qa = ptr(d[9]), *qas = ptr(d[10]), *kb = ptr(d[13]),
          *ks = ptr(d[14]);
  uint8_t *vb = ptr(d[17]), *vs = ptr(d[18]);
  float x[128];
  for (uint32_t t = 0; t < M; t++)
    for (uint32_t h = 0; h < hq + hkv; h++) {
      for (uint32_t c = 0; c < half; c++) {
        float lo = bf(qkv[t * nqkv + h * hd + c]),
              hi = bf(qkv[t * nqkv + h * hd + half + c]);
        x[c] = lo * cos[t * half + c] - hi * sin[t * half + c];
        x[half + c] = hi * cos[t * half + c] + lo * sin[t * half + c];
      }
      if (h < hq) {
        for (uint32_t g = 0; g < hd / 32; g++)
          quant_a(x + g * 32, qa + (h / gq) * d[11], qas + (h / gq) * d[12], tm,
                  hd, hd, (h % gq) * M + t, g);
      } else {
        uint32_t head = h - hq, key = d[21] + t, nt = key / 64, k = key % 64;
        for (uint32_t g = 0; g < hd / 32; g++)
          quant_group(x + g * 32,
                      kb + head * d[15] + nt * hd * 64 + g * 32 * 64 + k, 64,
                      ks + head * d[16] + nt * (hd / 32) * 64 + g * 64 + k);
      }
    }
  for (uint32_t h = 0; h < hkv; h++)
    for (uint32_t t0 = 0; t0 < M; t0 += 32)
      for (uint32_t c = 0; c < hd; c++) {
        for (uint32_t t = 0; t < 32; t++)
          x[t] = bf(qkv[(t0 + t) * nqkv + (hq + hkv) * hd + h * hd + c]);
        uint32_t key = d[21] + t0;
        quant_group(
            x,
            vb + h * d[19] + ((key / 256) * (hd / 64) + c / 64) * 256 * 64 +
                (key % 256) * 64 + c % 64,
            64,
            vs + h * d[20] + ((c / 64) * (SK / 256) + key / 256) * 8 * 64 +
                ((key % 256) / 32) * 64 + c % 64);
      }
}
static void softmax(const uint32_t *d) {
  uint16_t *s = ptr(d[2]);
  uint8_t *a = ptr(d[7]), *asc = ptr(d[8]);
  uint32_t SK = d[4], R = d[5], M = d[13], tm = d[14], tk = d[15];
  float scale = value(d[11]);
  for (uint32_t b = 0; b < d[6]; b++)
    for (uint32_t r = 0; r < R; r++) {
      uint16_t *sr = (uint16_t *)((unsigned char *)s + b * d[3]) + r * SK;
      uint32_t last = d[12] + r % M;
      float maximum = -INFINITY;
      for (uint32_t k = 0; k <= last; k++)
        maximum = __builtin_fmaxf(maximum, bf(sr[k]) * scale);
      for (uint32_t k = 0; k < SK; k++)
        exp_input[k] = bf16(bf(sr[k]) * scale - maximum);
      exp_bf16(exp_input, exp_output, SK);
      float sums[16] = {0};
      for (uint32_t lane = 0; lane < 16; lane++)
        for (uint32_t w = lane; w < SK / 2; w += 16) {
          float x = 2 * w <= last ? bf(exp_output[2 * w]) : 0;
          float y = 2 * w + 1 <= last ? bf(exp_output[2 * w + 1]) : 0;
          sums[lane] += x + y;
        }
      float sum = 0;
      for (unsigned lane = 0; lane < 16; lane++)
        sum += sums[lane];
      float inv = 1.0f / sum;
      for (uint32_t k = 0; k < SK; k++)
        row[k] = k <= last ? bf(exp_output[k]) * inv : 0;
      for (uint32_t g = 0; g < SK / 32; g++)
        quant_a(row + g * 32, a + b * d[9], asc + b * d[10], tm, tk, SK, r, g);
    }
}
static int guards(void) {
  unsigned char *g[] = {rk_guard1a, rk_guard1b, rk_guard2a,
                        rk_guard2b, rk_guard3a, rk_guard3b};
  int bad = 0;
  for (unsigned j = 0; j < 6; j++)
    for (unsigned k = 0; k < 64; k++)
      bad += g[j][k] != 0xa5;
  return bad;
}
int main(void) {
  const uint32_t *program = ptr(LL_PROG_ADDR);
  /* Refuse unsupported/fused programs before starting a timed execution. */
  for (uint32_t p = 0; p < LL_NP; p++) {
    const uint32_t *d = program + p * 32;
    if (d[0] < 1 || d[0] > 7 || d[0] == 6 || (d[0] == 1 && d[20]) ||
        (d[0] == 2 && d[13]) || (d[0] == 3 && d[25]))
      return 4;
  }
  uint64_t times[LL_NP];
  fence();
  uint64_t begin = cycle();
#ifdef RK_EMBEDDING_LOOKUP
  uint16_t *x0 = ptr(RK_X0_ADDRESS);
  for (uint32_t t = 0; t < LL_M; t++)
    for (uint32_t c = 0; c < LL_D; c++)
      x0[t * LL_D + c] = bf16(rk_embeddings[rk_token_ids[t] * LL_D + c]);
#endif
  for (uint32_t p = 0; p < LL_NP; p++) {
    const uint32_t *d = program + p * 32;
#if RK_PROFILE_PHASES
    uint64_t t = cycle();
#endif
    switch (d[0]) {
    case 1:
      gemm(d);
      break;
    case 2:
      prenorm(d);
      break;
    case 3:
      rope(d);
      break;
    case 4:
      softmax(d);
      break;
    case 5:
      ewq(d, 0);
      break;
    case 7:
      ewq(d, 1);
      break;
    }
    fence();
#if RK_PROFILE_PHASES
    times[p] = cycle() - t;
#endif
  }
#ifdef RK_EMBEDDING_LOOKUP
  const uint16_t *bf16_logits = ptr(LL_LOGITS_ADDR);
  for (uint32_t i = 0; i < LL_M * LL_V; i++)
    rk_logits_f32[i] = bf(bf16_logits[i]);
#endif
  fence();
  uint64_t total = cycle() - begin;
#if RK_PROFILE_PHASES
  for (uint32_t p = 0; p < LL_NP; p++)
    printf("PORT_PHASE %u %u %lu\n", p, program[p * 32 + 1],
           (unsigned long)times[p]);
#endif
  printf("PORT_PHASES_COMPLETED %u\n", LL_NP);
  printf("PORT_TOTAL %lu guards=%d\n", (unsigned long)total, guards());
  for (unsigned j = 0; j < RK_NCHECKS; j++) {
    const struct RkCheck *check = RK_CHECKS + j;
    uint8_t *src;
#ifdef RK_EMBEDDING_LOOKUP
    src = check->address == 0 ? (uint8_t *)rk_logits_f32 : ptr(check->address);
#else
    src = ptr(check->address);
#endif
    for (uint32_t offset = 0; offset < check->bytes; offset += 32) {
      uint32_t count = check->bytes - offset;
      if (count > 32)
        count = 32;
      printf("PORT_BYTES %u %u %u", j, offset, count);
      for (unsigned k = 0; k < count; k++)
        printf(" %02x", src[offset + k]);
      printf("\n");
    }
  }
  printf("PORT_DONE\n");
  return guards() ? 3 : 0;
}
#endif
