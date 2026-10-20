/* Continuous cached execution of the Radiance-derived Rocket kernels.
 * Descriptor construction is independent of the compiler's host graph. */
#define main source_forward_main
#include "radiance_rocket_llama.c"
#undef main

extern const uint32_t rk_session_programs[];
extern const float rk_session_cos[], rk_session_sin[], rk_embeddings[];
extern const int64_t rk_token_ids[];
static float session_logits[RK_STEPS * LL_V];
/* Preserve unquantized cache values so a partially filled 32-token V group
 * can be requantized when a new token changes its shared exponent. */
static struct {
  uint64_t before;
  float values[2][LL_L][LL_HKV][RK_CAPACITY][LL_HD];
  uint64_t after;
} cache;

static void cached_rope(const uint32_t *d) {
  const uint16_t *qkv = ptr(d[2]);
  unsigned layer = d[1] >> 8, nqkv = d[3], hq = d[4], hkv = d[5];
  unsigned hd = d[6], half = hd / 2, M = d[22], SK = d[23];
  unsigned tm = d[24], gq = hq / hkv, p0 = d[21];
  uint8_t *qa = ptr(d[9]), *qas = ptr(d[10]);
  uint8_t *kb = ptr(d[13]), *ks = ptr(d[14]);
  uint8_t *vb = ptr(d[17]), *vs = ptr(d[18]);
  float x[128];
  for (unsigned t = 0; t < M; t++) {
    const float *cs = rk_session_cos + (p0 + t) * half;
    const float *sn = rk_session_sin + (p0 + t) * half;
    for (unsigned h = 0; h < hq + hkv; h++) {
      for (unsigned c = 0; c < half; c++) {
        float lo = bf(qkv[t * nqkv + h * hd + c]);
        float hi = bf(qkv[t * nqkv + h * hd + half + c]);
        x[c] = lo * cs[c] - hi * sn[c];
        x[half + c] = hi * cs[c] + lo * sn[c];
      }
      if (h < hq) {
        for (unsigned g = 0; g < hd / 32; g++)
          quant_a(x + g * 32, qa + (h / gq) * d[11],
                  qas + (h / gq) * d[12], tm, hd, hd,
                  (h % gq) * M + t, g);
      } else {
        unsigned head = h - hq, key = p0 + t;
        for (unsigned c = 0; c < hd; c++)
          cache.values[0][layer][head][key][c] = x[c];
        for (unsigned g = 0; g < hd / 32; g++)
          quant_group(x + g * 32,
                      kb + head * d[15] + (key / 64) * hd * 64 + g * 32 * 64 + key % 64,
                      64, ks + head * d[16] + (key / 64) * (hd / 32) * 64 + g * 64 + key % 64);
      }
    }
    for (unsigned h = 0; h < hkv; h++)
      for (unsigned c = 0; c < hd; c++)
        cache.values[1][layer][h][p0 + t][c] =
            bf(qkv[t * nqkv + (hq + hkv) * hd + h * hd + c]);
  }
  for (unsigned h = 0; h < hkv; h++)
    for (unsigned key = (p0 / 32) * 32; key < p0 + M; key += 32)
      for (unsigned c = 0; c < hd; c++) {
        for (unsigned t = 0; t < 32; t++)
          x[t] = key + t < RK_CAPACITY ? cache.values[1][layer][h][key + t][c] : 0;
        quant_group(x,
                    vb + h * d[19] + ((key / 256) * (hd / 64) + c / 64) * 256 * 64 +
                        (key % 256) * 64 + c % 64,
                    64,
                    vs + h * d[20] + ((c / 64) * (SK / 256) + key / 256) * 8 * 64 +
                        ((key % 256) / 32) * 64 + c % 64);
      }
}

static void dump(unsigned index, const void *data, unsigned bytes) {
  const uint8_t *src = data;
  for (unsigned offset = 0; offset < bytes; offset += 32) {
    unsigned count = bytes - offset;
    if (count > 32) count = 32;
    printf("PORT_BYTES %u %u %u", index, offset, count);
    for (unsigned k = 0; k < count; k++) printf(" %02x", src[offset + k]);
    printf("\n");
  }
}

int main(void) {
  uint64_t total = 0, step_cycles[RK_STEPS];
  fence();
  uint64_t begin = cycle();
  cache.before = cache.after = 0x5a39c8e17d20b64fULL;
  for (unsigned i = 0; i < sizeof(cache.values) / sizeof(float); i++)
    ((float *)cache.values)[i] = 0;
  /* Reset the persistent packed cache explicitly inside the measured interval. */
  for (unsigned p = 0; p < LL_NP; p++) {
    const uint32_t *d = rk_session_programs + p * 32;
    if (d[0] != 3) continue;
    for (unsigned h = 0; h < LL_HKV; h++) {
      for (unsigned i = 0; i < d[15]; i++) ((uint8_t *)ptr(d[13]))[h * d[15] + i] = 0;
      for (unsigned i = 0; i < d[16]; i++) ((uint8_t *)ptr(d[14]))[h * d[16] + i] = 104;
      for (unsigned i = 0; i < d[19]; i++) ((uint8_t *)ptr(d[17]))[h * d[19] + i] = 0;
      for (unsigned i = 0; i < d[20]; i++) ((uint8_t *)ptr(d[18]))[h * d[20] + i] = 104;
    }
  }
  fence();
  uint64_t reset_cycles = cycle() - begin;
  total += reset_cycles;
  for (unsigned step = 0; step < RK_STEPS; step++) {
    fence();
    begin = cycle();
    const uint32_t *program = rk_session_programs + step * LL_NP * 32;
    unsigned rows = step ? 1 : LL_M;
    uint16_t *x0 = ptr(program[2]);
    if (step || RK_SESSION_RECORDED)
      for (unsigned t = 0; t < rows; t++)
        for (unsigned c = 0; c < LL_D; c++)
          x0[t * LL_D + c] = bf16(rk_embeddings[rk_token_ids[step ? LL_M + step - 1 : t] * LL_D + c]);
    for (unsigned p = 0; p < LL_NP; p++) {
      const uint32_t *d = program + p * 32;
      switch (d[0]) {
      case 1: gemm(d); break;
      case 2: prenorm(d); break;
      case 3: cached_rope(d); break;
      case 4: softmax(d); break;
      case 5: ewq(d, 0); break;
      case 7: ewq(d, 1); break;
      default: return 4;
      }
      fence();
    }
    const uint16_t *logits = ptr(LL_LOGITS_ADDR);
    for (unsigned c = 0; c < LL_V; c++)
      session_logits[step * LL_V + c] = bf(logits[(rows - 1) * LL_V + c]);
    fence();
    step_cycles[step] = cycle() - begin;
    total += step_cycles[step];
  }
  int bad = guards() + (cache.before != 0x5a39c8e17d20b64fULL) +
            (cache.after != 0x5a39c8e17d20b64fULL);
  printf("CACHED_RESET %lu\n", (unsigned long)reset_cycles);
  for (unsigned step = 0; step < RK_STEPS; step++)
    printf("CACHED_STEP %u %u %lu\n", step, LL_M + step, (unsigned long)step_cycles[step]);
  printf("PORT_PHASES_COMPLETED %u\n", LL_NP * RK_STEPS);
  printf("PORT_TOTAL %lu guards=%d\n", (unsigned long)total, bad);
  dump(0, session_logits, sizeof(session_logits));
  dump(1, cache.values, sizeof(cache.values));
  printf("PORT_DONE\n");
  return bad ? 3 : 0;
}
