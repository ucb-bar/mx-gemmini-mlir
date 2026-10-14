/* Check a cold compiler head before the paired source/issuer timing. */
#define ATTN_HEADER "smol_head0.h"
#define ATTN_EXPECT "smol_expect.h"
#define ATTN_SERIAL 1
#define main source_main
#include "attn_flash_smol.c"
#undef main

extern void mx_issue(const void *, const void *, const void *, const void *,
                     const void *, const void *, const void *, const void *,
                     const void *, const void *, const void *);
static uint16_t compiler_first[SQ][D] __attribute__((aligned(64)));
static uint16_t source_reference[SQ][D] __attribute__((aligned(64)));

static void issue_compiler(void) {
  mx_issue(q_scales_folded, ATTN_KTS, ATTN_VS, ATTN_Q, ATTN_KT,
           ATTN_V, scale_sink, p_scales[0], p_scales[1], O_hw,
           MASK_BLOCK_BF16);
}

int main(void) {
  for (int g = 0; g < D / 32; g++)
    for (int m = 0; m < SQ; m++)
      q_scales_folded[g][m] = ATTN_QS[g][m] - FOLD_SHIFT;
  gemmini_flush(0);
  gemmini_fence();
  issue_compiler();
  gemmini_fence();
  if (fnv(O_hw, sizeof(O_hw)) != EXP_FLASH_HASH_O) return 3;
  memcpy(compiler_first, O_hw, sizeof(O_hw));
  printf("compiler first fresh guest hash %016llx\n",
         (unsigned long long)fnv(O_hw, sizeof(O_hw)));

  if (source_main()) return 1;
  const uint64_t expected = fnv(O_hw, sizeof(O_hw));
  memcpy(source_reference, O_hw, sizeof(O_hw));
  for (int word = 0; word < SQ * D; word++)
    if (((volatile uint16_t *)compiler_first)[word] !=
        ((volatile uint16_t *)source_reference)[word]) return 3;
  printf("compiler first full BF16 output match %d values\n", SQ * D);

  for (int trial = 0; trial < 4; trial++) {
    gemmini_flush(0);
    gemmini_fence();
    uint64_t cycles;
    if ((trial & 1) == 0) {
      cycles = run_pipelined();
      printf("paired pipeline %d cycles %llu hash %016llx\n", trial,
             (unsigned long long)cycles,
             (unsigned long long)fnv(O_hw, sizeof(O_hw)));
    } else {
      uint64_t start = read_cycles();
      issue_compiler();
      gemmini_fence();
      cycles = read_cycles() - start;
      printf("paired compiler %d cycles %llu hash %016llx\n", trial,
             (unsigned long long)cycles,
             (unsigned long long)fnv(O_hw, sizeof(O_hw)));
    }
    if (fnv(O_hw, sizeof(O_hw)) != expected) return 2;
    for (int word = 0; word < SQ * D; word++)
      if (((volatile uint16_t *)O_hw)[word] !=
          ((volatile uint16_t *)source_reference)[word]) return 2;
  }
  printf("paired full BF16 output match %d values\n", SQ * D);
  return 0;
}
