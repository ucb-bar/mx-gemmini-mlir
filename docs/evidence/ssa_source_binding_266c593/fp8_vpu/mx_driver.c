#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[];
extern const uint8_t activation_scales[];
extern const uint8_t golden_bf16[];
extern const uint8_t golden_fp8[];
extern const uint8_t golden_output_scales[];
extern const uint8_t nicolas_fp8[];
extern const uint8_t nicolas_output_scales[];
extern const uint8_t output_scales[];
extern const uint8_t weight[];
extern const uint8_t weight_scales[];

static uint8_t output_quantized[4096] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));

void mx_issue(const void *activation, const void *activation_scales, const void *output_quantized, const void *scratch_output_scales, const void *weight, const void *weight_scales);

int main(void) {
  mx_issue(activation, activation_scales, output_quantized, scratch_output_scales, weight, weight_scales);
  int code_errors = 0;
  int scale_errors = 0;
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {
      uint32_t tiled = (((row / 16) * (64 / 16) + col / 16) * 16 + row % 16) * 16 + col % 16;
      uint32_t linear = row * 64 + col;
      if (output_quantized[tiled] != nicolas_fp8[linear]) {
        if (code_errors < 8)
          printf("CODE MISMATCH (%u,%u): got=0x%02x expected=0x%02x\n",
                 row, col, output_quantized[tiled], nicolas_fp8[linear]);
        ++code_errors;
      }
    }
  for (uint32_t i = 0; i < 128; ++i) {
    if (scratch_output_scales[i] != nicolas_output_scales[i]) {
      if (scale_errors < 8)
        printf("SCALE MISMATCH %u: got=0x%02x expected=0x%02x\n",
               i, scratch_output_scales[i], nicolas_output_scales[i]);
      ++scale_errors;
    }
  }
  printf("lowered MX 64x64x128: %d FP8 code mismatches, %d E8M0 scale mismatches\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}
