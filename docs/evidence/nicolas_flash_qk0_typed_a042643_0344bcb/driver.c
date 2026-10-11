/* Source oracle and compiler-issued QK0 check for attn_flash_mid.c.
 * The source path calls the original load_q/load_k/issue_mm functions.
 * GENERATED calls only mx_issue; both print every BF16 score word.
 */
#define main source_main
#include "attn_flash_mid.c"
#undef main

static uint16_t qk0_scores[SQ][BK] __attribute__((aligned(64)));

#ifdef GENERATED
extern void mx_issue(const void *, const void *, const void *, const void *,
                     const void *, const void *);
#endif

int main(void) {
  gemmini_flush(0);
#ifdef GENERATED
  for (int g = 0; g < D / 32; g++)
    for (int m = 0; m < SQ; m++)
      q_scales_folded[g][m] = ATTN_QS[g][m] - FOLD_SHIFT;
  mx_issue(q_scales_folded, ATTN_KTS, ATTN_Q, ATTN_KT,
           scale_sink, qk0_scores);
#else
  attn_load_q_scales();
  load_q();
  load_k(0);
  n_mm = 1;
  mm_next = 0;
  sched[0] = (mm_t){0, 0};
  issue_mm();
  mvout_rows(qk0_scores, S_BUF(0), ROWS16(SQ, BK));
#endif
  gemmini_fence();
  printf("QK0 score hash %016llx\n",
         (unsigned long long)fnv(qk0_scores, sizeof(qk0_scores)));
  printf("QK0 BF16 words ");
  for (int i = 0; i < SQ * BK; i++)
    printf("%04x", ((uint16_t *)qk0_scores)[i]);
  printf("\n");
  return 0;
}
