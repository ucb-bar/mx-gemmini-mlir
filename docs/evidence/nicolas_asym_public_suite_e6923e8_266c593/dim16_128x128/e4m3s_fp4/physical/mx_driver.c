#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[];
extern const uint8_t activation_scales[];
extern const uint8_t golden_bf16[];
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
  for (uint32_t i = 0; i < 16384; ++i) {
    if (got[i] != expected[i]) {
      if (errors < 8)
        printf("MISMATCH %u: got=0x%04x expected=0x%04x\n",
               i, got[i], expected[i]);
      ++errors;
    }
  }
  printf("lowered MX 128x128x128: %d BF16 mismatches\n", errors);
  return errors != 0;
}
