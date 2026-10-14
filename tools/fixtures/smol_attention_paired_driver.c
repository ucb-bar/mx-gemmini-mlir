/* Compare one captured SmolVLA source head with a compiler-issued head. */
#define ATTN_HEADER "smol_head0.h"
#define ATTN_EXPECT "smol_expect.h"
#define ATTN_SERIAL 1
#define main source_main
#include "attn_flash_smol.c"
#undef main

extern void mx_issue(const void *, const void *, const void *, const void *,
                     const void *, const void *, const void *, const void *,
                     const void *, const void *, const void *);
static uint16_t source_reference[SQ][D] __attribute__((aligned(64)));

int main(void) {
  if (source_main()) return 1;
  const uint64_t expected = fnv(O_hw, sizeof(O_hw));
  memcpy(source_reference, O_hw, sizeof(O_hw));
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
      mx_issue(q_scales_folded, ATTN_KTS, ATTN_VS, ATTN_Q, ATTN_KT,
               ATTN_V, scale_sink, p_scales[0], p_scales[1], O_hw,
               MASK_BLOCK_BF16);
      gemmini_fence();
      cycles = read_cycles() - start;
      printf("paired compiler %d cycles %llu hash %016llx\n", trial,
             (unsigned long long)cycles,
             (unsigned long long)fnv(O_hw, sizeof(O_hw)));
    }
    if (fnv(O_hw, sizeof(O_hw)) != expected) return 2;
    for (int word = 0; word < SQ * D; word++) {
      if (((volatile uint16_t *)O_hw)[word] !=
          ((volatile uint16_t *)source_reference)[word]) return 2;
    }
  }
  printf("paired full BF16 output match %d values\n", SQ * D);
  return 0;
}
