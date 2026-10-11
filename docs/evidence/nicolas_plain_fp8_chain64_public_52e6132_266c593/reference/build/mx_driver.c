#include <stdint.h>
#include <stdio.h>
extern const uint8_t a1_activation[];
extern const uint8_t a1_scales[];
extern const uint8_t b1_scales[];
extern const uint8_t b1_weight[];
extern const uint8_t b2_scales[];
extern const uint8_t b2_weight[];
extern const uint8_t c1_codes_ref[];
extern const uint8_t c1_scales_ref[];
extern const uint8_t c2_codes_ref[];
extern const uint8_t c2_scales_ref[];
static uint8_t c1_scales[128] __attribute__((aligned(64)));
static uint8_t c1_tiled_observed[4096] __attribute__((aligned(64)));
static uint8_t c2_scales[128] __attribute__((aligned(64)));
static uint8_t c2_tiled[4096] __attribute__((aligned(64)));
void mx_issue(const void *a1_activation, const void *a1_scales, const void *b1_scales, const void *b1_weight, const void *b2_scales, const void *b2_weight, const void *c1_scales, const void *c1_tiled_observed, const void *c2_scales, const void *c2_tiled);

int main(void) {
  mx_issue(a1_activation, a1_scales, b1_scales, b1_weight, b2_scales, b2_weight, c1_scales, c1_tiled_observed, c2_scales, c2_tiled);
  int c1_codes = 0, c1_scale_errors = 0, c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c1_codes += c1_tiled_observed[tiled] != c1_codes_ref[row * 64 + col];
      c2_codes += c2_tiled[tiled] != c2_codes_ref[row * 64 + col];
    }
  for (uint32_t i = 0; i < 128; ++i) {
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  }
  printf("lowered connected 64x64: C1 %d codes %d scales; C2 %d codes %d scales\n",
         c1_codes, c1_scale_errors, c2_codes, c2_scale_errors);
  return c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}
