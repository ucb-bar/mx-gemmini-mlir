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

static uint8_t output_bf16[32768] __attribute__((aligned(64)));
static uint8_t output_quantized[16384] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[512] __attribute__((aligned(64)));

void mx_issue(const void *activation, const void *activation_scales, const void *output_bf16, const void *scratch_output_scales, const void *weight, const void *weight_scales);

static float bf16_value(uint16_t bits) {
  union { uint32_t u; float f; } v = { .u = (uint32_t)bits << 16 };
  return v.f;
}

static uint32_t float_bits(float value) {
  union { float f; uint32_t u; } v = { .f = value };
  return v.u;
}

static float power_of_two(int exponent) {
  union { uint32_t u; float f; } v = {
    .u = exponent >= -126 ? (uint32_t)(exponent + 127) << 23 :
         (uint32_t)1 << (exponent + 149)
  };
  return v.f;
}

static int floor_log2_positive_bits(uint32_t bits) {
  int biased = (int)((bits >> 23) & 255);
  if (biased) return biased - 127;
  uint32_t fraction = bits & 0x7fffff;
  int leading = -1;
  while (fraction) { fraction >>= 1; ++leading; }
  return leading - 149;
}

static int rne_integer(float x) {
  int whole = (int)x;
  float fraction = x - (float)whole;
  return fraction < 0.5f ? whole : fraction > 0.5f ? whole + 1 :
         (whole & 1) ? whole + 1 : whole;
}

static uint8_t radiance_fp8_code(float x) {
  uint32_t raw = float_bits(x);
  if ((raw & 0x7fffffff) == 0 || ((raw >> 23) & 255) == 255) return 0;
  int sign = (raw >> 24) & 0x80;
  float magnitude = x < 0.0f ? -x : x;
  int exponent = floor_log2_positive_bits(raw & 0x7fffffff);
  if (exponent < -6) return 0;
  int mantissa;
  if (exponent > 8) { exponent = 8; mantissa = 6; }
  else {
    float base = power_of_two(exponent);
    mantissa = rne_integer((magnitude - base) / (base / 8.0f));
    if (mantissa >= 8) {
      ++exponent; mantissa = 0;
      if (exponent > 8) { exponent = 8; mantissa = 6; }
    } else {
      int high = exponent == 8 ? 6 : 7;
      if (mantissa > high) mantissa = high;
      if (mantissa < 0) mantissa = 0;
    }
  }
  return (uint8_t)(sign | (((exponent + 7) & 15) << 3) | mantissa);
}

static void radiance_header_requantize(void) {
  const uint16_t *input = (const uint16_t *)output_bf16;
  for (uint32_t row = 0; row < 128; ++row)
    for (uint32_t group = 0; group < 4; ++group) {
      uint32_t begin = row * 128 + group * 32;
      uint16_t maximum = 0;
      for (uint32_t offset = 0; offset < 32; ++offset) {
        uint16_t magnitude = input[begin + offset] & 0x7fff;
        if (magnitude > maximum) maximum = magnitude;
      }
      int exponent = maximum == 0 ? 0 :
          floor_log2_positive_bits((uint32_t)maximum << 16) - 8 + 127;
      uint8_t scale_code = maximum == 0 ? 0 :
          (uint8_t)(exponent < 0 ? 0 : exponent > 254 ? 254 : exponent);
      scratch_output_scales[row * 4 + group] = scale_code;
      float scale = power_of_two((int)scale_code - 127);
      for (uint32_t offset = 0; offset < 32; ++offset)
        output_quantized[begin + offset] =
            radiance_fp8_code(bf16_value(input[begin + offset]) / scale);
    }
}

int main(void) {
  mx_issue(activation, activation_scales, output_bf16, scratch_output_scales, weight, weight_scales);
  radiance_header_requantize();
  int code_errors = 0, scale_errors = 0;
  for (uint32_t i = 0; i < 16384; ++i)
    if (output_quantized[i] != golden_fp8[i]) ++code_errors;
  for (uint32_t i = 0; i < 512; ++i)
    if (scratch_output_scales[i] != golden_output_scales[i]) ++scale_errors;
  printf("lowered MX 128x128x256: %d Radiance FP8 code mismatches, %d E8M0 scale mismatches\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}
