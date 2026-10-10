#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
extern const uint8_t first_activation[];
extern const uint8_t first_activation_scales[];
extern const uint8_t first_weight[];
extern const uint8_t first_weight_scales[];
extern const uint8_t first_golden_bf16[];
extern const uint8_t second_activation[];
extern const uint8_t second_activation_scales[];
extern const uint8_t second_weight[];
extern const uint8_t second_weight_scales[];
extern const uint8_t second_golden_bf16[];

static uint8_t first_output[8192] __attribute__((aligned(64)));
static uint8_t second_output[8192] __attribute__((aligned(64)));
static uint8_t first_scratch[2048] __attribute__((aligned(64)));
static uint8_t second_scratch[2048] __attribute__((aligned(64)));

static int compare(const uint8_t *actual_bytes, const uint8_t *expected_bytes,
                   const char *label) {
  const uint16_t *actual = (const uint16_t *)actual_bytes;
  const uint16_t *expected = (const uint16_t *)expected_bytes;
  int errors = 0;
  for (int i = 0; i < 4096; ++i) {
    if (actual[i] != expected[i]) {
      if (errors < 8)
        printf("%s mismatch %d: got=0x%04x expected=0x%04x\n",
               label, i, actual[i], expected[i]);
      ++errors;
    }
  }
  return errors;
}

int main(void) {
  mx_issue(first_activation, first_activation_scales, first_output,
           first_scratch, first_weight, first_weight_scales);
  int first_errors = compare(first_output, first_golden_bf16, "first");
  mx_issue(second_activation, second_activation_scales, second_output,
           second_scratch, second_weight, second_weight_scales);
  int second_errors = compare(second_output, second_golden_bf16, "second");
  first_errors += compare(first_output, first_golden_bf16, "first_after_second");
  printf("runtime MX QK: %d/8192 BF16 mismatches\n",
         first_errors + second_errors);
  return first_errors || second_errors;
}
