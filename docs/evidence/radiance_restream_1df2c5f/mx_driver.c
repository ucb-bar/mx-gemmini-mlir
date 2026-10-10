#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[];
extern const uint8_t activation_scales[];
extern const uint8_t golden_bf16[];
extern const uint8_t output_scales[];
extern const uint8_t weight[];
extern const uint8_t weight_scales[];

static uint8_t output_bf16[32768] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));

void mx_issue(const void *activation, const void *activation_scales, const void *output_bf16, const void *scratch_output_scales, const void *weight, const void *weight_scales);

int main(void) {
  mx_issue(activation, activation_scales, output_bf16, scratch_output_scales, weight, weight_scales);
  const uint16_t *got = (const uint16_t *)output_bf16;
  const uint16_t *expected = (const uint16_t *)golden_bf16;
  int errors = 0;
  for (uint32_t row = 0; row < 256; ++row)
    for (uint32_t col = 0; col < 64; ++col) {
      uint32_t tile = (row / 64) * 1 + col / 64;
      uint32_t local = (row % 64) * 64 + col % 64;
      uint32_t got_index = tile * 4096 + local;
      uint32_t expected_index = row * 64 + col;
      if (got[got_index] != expected[expected_index]) {
        if (errors < 8)
          printf("MISMATCH (%u,%u): got=0x%04x expected=0x%04x\n",
                 row, col, got[got_index], expected[expected_index]);
        ++errors;
      }
    }
  printf("lowered MX 256x64x2048: %d BF16 mismatches\n", errors);
  return errors != 0;
}
