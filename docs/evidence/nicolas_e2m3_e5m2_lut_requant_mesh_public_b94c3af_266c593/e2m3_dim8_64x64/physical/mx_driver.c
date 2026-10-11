#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[];
extern const uint8_t activation_lut[];
extern const uint8_t activation_scales[];
extern const uint8_t golden_bf16[];
extern const uint8_t golden_lut_indices[];
extern const uint8_t golden_output_scales[];
extern const uint8_t output_lut[];
extern const uint8_t weight[];
extern const uint8_t weight_lut[];
extern const uint8_t weight_scales[];

static uint8_t output_quantized[2048] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));

void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_lut, const void *output_quantized, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales);

int main(void) {
  mx_issue(activation, activation_lut, activation_scales, output_lut, output_quantized, scratch_output_scales, weight, weight_lut, weight_scales);
  int code_errors = 0;
  int scale_errors = 0;
  for (uint32_t i = 0; i < 2048; ++i) {
    if (output_quantized[i] != golden_lut_indices[i]) {
      if (code_errors < 8)
        printf("CODE MISMATCH %u: got=0x%02x expected=0x%02x\n",
               i, output_quantized[i], golden_lut_indices[i]);
      ++code_errors;
    }
  }
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t got_index = row * 2 + group;
      uint32_t expected_index = group * 64 + row;
      if (scratch_output_scales[got_index] != golden_output_scales[expected_index]) {
        if (scale_errors < 8)
          printf("SCALE MISMATCH (%u,%u): got=0x%02x expected=0x%02x\n",
                 row, group, scratch_output_scales[got_index],
                 golden_output_scales[expected_index]);
        ++scale_errors;
      }
    }

  printf("lowered MX 64x64x64: %d E2M3 packed-LUT-index mismatches, %d E8M0 scale mismatches\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}
