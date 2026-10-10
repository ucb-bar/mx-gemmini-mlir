#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
extern const uint8_t first_activation[];
extern const uint8_t first_activation_scales[];
extern const uint8_t first_weight[];
extern const uint8_t first_weight_scales[];
extern const uint8_t first_expected_bf16[];
extern const uint8_t second_activation[];
extern const uint8_t second_activation_scales[];
extern const uint8_t second_weight[];
extern const uint8_t second_weight_scales[];
extern const uint8_t second_expected_bf16[];

static uint8_t first_output[131072] __attribute__((aligned(64)));
static uint8_t second_output[131072] __attribute__((aligned(64)));
static uint8_t first_scratch[2048] __attribute__((aligned(64)));
static uint8_t second_scratch[2048] __attribute__((aligned(64)));

static int compare(const uint8_t *actual_bytes, const uint8_t *expected_bytes,
                   const char *label) {
  const uint16_t *actual = (const uint16_t *)actual_bytes;
  const uint16_t *expected = (const uint16_t *)expected_bytes;
  int errors = 0;
  for (int i = 0; i < 65536; ++i) {
    const int row = i / 256;
    const int col = i % 256;
    const int tile = (row / 128) * 2 + col / 128;
    const int local = (row % 128) * 128 + col % 128;
    const int physical = tile * 16384 + local;
    if (actual[physical] != expected[i]) {
      if (errors < 8)
        printf("%s mismatch %d: got=0x%04x expected=0x%04x\n",
               label, i, actual[physical], expected[i]);
      ++errors;
    }
  }
  return errors;
}

int main(void) {
  mx_issue(first_activation, first_activation_scales, first_output,
           first_scratch, first_weight, first_weight_scales);
  int first_errors = compare(first_output, first_expected_bf16, "first");
  mx_issue(second_activation, second_activation_scales, second_output,
           second_scratch, second_weight, second_weight_scales);
  int second_errors = compare(second_output, second_expected_bf16, "second");
  first_errors += compare(first_output, first_expected_bf16, "first_after_second");
  printf("runtime FP4 tilewise: %d/131072 BF16 mismatches\n",
         first_errors + second_errors);
  return first_errors || second_errors;
}
