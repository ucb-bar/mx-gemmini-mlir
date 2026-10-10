#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
extern const uint8_t tile0_activation[];
extern const uint8_t tile0_activation_scales[];
extern const uint8_t tile0_weight[];
extern const uint8_t tile0_weight_scales[];
extern const uint8_t tile0_golden_bf16[];
extern const uint8_t tile1_activation[];
extern const uint8_t tile1_activation_scales[];
extern const uint8_t tile1_weight[];
extern const uint8_t tile1_weight_scales[];
extern const uint8_t tile1_golden_bf16[];
extern const uint8_t tile2_activation[];
extern const uint8_t tile2_activation_scales[];
extern const uint8_t tile2_weight[];
extern const uint8_t tile2_weight_scales[];
extern const uint8_t tile2_golden_bf16[];
extern const uint8_t tile3_activation[];
extern const uint8_t tile3_activation_scales[];
extern const uint8_t tile3_weight[];
extern const uint8_t tile3_weight_scales[];
extern const uint8_t tile3_golden_bf16[];
extern const uint8_t tile4_activation[];
extern const uint8_t tile4_activation_scales[];
extern const uint8_t tile4_weight[];
extern const uint8_t tile4_weight_scales[];
extern const uint8_t tile4_golden_bf16[];
extern const uint8_t tile5_activation[];
extern const uint8_t tile5_activation_scales[];
extern const uint8_t tile5_weight[];
extern const uint8_t tile5_weight_scales[];
extern const uint8_t tile5_golden_bf16[];
extern const uint8_t tile6_activation[];
extern const uint8_t tile6_activation_scales[];
extern const uint8_t tile6_weight[];
extern const uint8_t tile6_weight_scales[];
extern const uint8_t tile6_golden_bf16[];
extern const uint8_t tile7_activation[];
extern const uint8_t tile7_activation_scales[];
extern const uint8_t tile7_weight[];
extern const uint8_t tile7_weight_scales[];
extern const uint8_t tile7_golden_bf16[];
extern const uint8_t tile8_activation[];
extern const uint8_t tile8_activation_scales[];
extern const uint8_t tile8_weight[];
extern const uint8_t tile8_weight_scales[];
extern const uint8_t tile8_golden_bf16[];
extern const uint8_t tile9_activation[];
extern const uint8_t tile9_activation_scales[];
extern const uint8_t tile9_weight[];
extern const uint8_t tile9_weight_scales[];
extern const uint8_t tile9_golden_bf16[];
extern const uint8_t tile10_activation[];
extern const uint8_t tile10_activation_scales[];
extern const uint8_t tile10_weight[];
extern const uint8_t tile10_weight_scales[];
extern const uint8_t tile10_golden_bf16[];
extern const uint8_t tile11_activation[];
extern const uint8_t tile11_activation_scales[];
extern const uint8_t tile11_weight[];
extern const uint8_t tile11_weight_scales[];
extern const uint8_t tile11_golden_bf16[];
extern const uint8_t tile12_activation[];
extern const uint8_t tile12_activation_scales[];
extern const uint8_t tile12_weight[];
extern const uint8_t tile12_weight_scales[];
extern const uint8_t tile12_golden_bf16[];
extern const uint8_t tile13_activation[];
extern const uint8_t tile13_activation_scales[];
extern const uint8_t tile13_weight[];
extern const uint8_t tile13_weight_scales[];
extern const uint8_t tile13_golden_bf16[];
extern const uint8_t tile14_activation[];
extern const uint8_t tile14_activation_scales[];
extern const uint8_t tile14_weight[];
extern const uint8_t tile14_weight_scales[];
extern const uint8_t tile14_golden_bf16[];
extern const uint8_t tile15_activation[];
extern const uint8_t tile15_activation_scales[];
extern const uint8_t tile15_weight[];
extern const uint8_t tile15_weight_scales[];
extern const uint8_t tile15_golden_bf16[];

static const uint8_t *const activation[16] = {tile0_activation, tile1_activation, tile2_activation, tile3_activation, tile4_activation, tile5_activation, tile6_activation, tile7_activation, tile8_activation, tile9_activation, tile10_activation, tile11_activation, tile12_activation, tile13_activation, tile14_activation, tile15_activation};
static const uint8_t *const activation_scales[16] = {tile0_activation_scales, tile1_activation_scales, tile2_activation_scales, tile3_activation_scales, tile4_activation_scales, tile5_activation_scales, tile6_activation_scales, tile7_activation_scales, tile8_activation_scales, tile9_activation_scales, tile10_activation_scales, tile11_activation_scales, tile12_activation_scales, tile13_activation_scales, tile14_activation_scales, tile15_activation_scales};
static const uint8_t *const weight[16] = {tile0_weight, tile1_weight, tile2_weight, tile3_weight, tile4_weight, tile5_weight, tile6_weight, tile7_weight, tile8_weight, tile9_weight, tile10_weight, tile11_weight, tile12_weight, tile13_weight, tile14_weight, tile15_weight};
static const uint8_t *const weight_scales[16] = {tile0_weight_scales, tile1_weight_scales, tile2_weight_scales, tile3_weight_scales, tile4_weight_scales, tile5_weight_scales, tile6_weight_scales, tile7_weight_scales, tile8_weight_scales, tile9_weight_scales, tile10_weight_scales, tile11_weight_scales, tile12_weight_scales, tile13_weight_scales, tile14_weight_scales, tile15_weight_scales};
static const uint8_t *const golden_bf16[16] = {tile0_golden_bf16, tile1_golden_bf16, tile2_golden_bf16, tile3_golden_bf16, tile4_golden_bf16, tile5_golden_bf16, tile6_golden_bf16, tile7_golden_bf16, tile8_golden_bf16, tile9_golden_bf16, tile10_golden_bf16, tile11_golden_bf16, tile12_golden_bf16, tile13_golden_bf16, tile14_golden_bf16, tile15_golden_bf16};

static uint8_t output[8192] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));

int main(void) {
  int errors = 0;
  for (int tile = 0; tile < 16; ++tile) {
    mx_issue(activation[tile], activation_scales[tile], output,
             scratch, weight[tile], weight_scales[tile]);
    const uint16_t *actual = (const uint16_t *)output;
    const uint16_t *expected = (const uint16_t *)golden_bf16[tile];
    for (int i = 0; i < 4096; ++i) {
      if (actual[i] != expected[i]) {
        if (errors < 8)
          printf("tile %d mismatch %d: got=0x%04x expected=0x%04x\n",
                 tile, i, actual[i], expected[i]);
        ++errors;
      }
    }
  }
  printf("runtime MX PV roster: %d/65536 BF16 mismatches\n", errors);
  return errors != 0;
}
