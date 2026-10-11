/* Compare the canonical streamed source with compiler-issued serial commands. */
#define main source_main
#include "attn_flash_llama.c"
#undef main

#ifdef GENERATED
extern void mx_issue(const void *, const void *, const void *, const void *,
                     const void *, const void *, const void *, const void *,
                     const void *, const void *);
#endif

int main(void) {
#ifdef GENERATED
  gemmini_flush(0);
  for (int g = 0; g < D / 32; g++)
    for (int m = 0; m < SQ; m++)
      q_scales_folded[g][m] = ATTN_QS[g][m] - FOLD_SHIFT;
  mx_issue(q_scales_folded, ATTN_KTS, ATTN_VS, ATTN_Q, ATTN_KT,
           ATTN_V, scale_sink, p_scales[0], p_scales[1], O_hw);
  gemmini_fence();
#else
  if (source_main()) return 1;
#endif
  uint64_t hash = fnv(O_hw, sizeof(O_hw));
  printf("O hash %016llx\n", (unsigned long long)hash);
  printf("O BF16 words ");
  for (int i = 0; i < SQ * D; i++)
    printf("%04x", ((uint16_t *)O_hw)[i]);
  printf("\n");
  return hash != EXP_FLASH_HASH_O;
}
