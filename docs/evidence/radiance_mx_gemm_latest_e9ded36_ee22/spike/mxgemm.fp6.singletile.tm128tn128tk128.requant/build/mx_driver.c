#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[];
extern const uint8_t activation_lut[];
extern const uint8_t activation_scales[];
extern const uint8_t golden_bf16[];
extern const uint8_t golden_output_scales[];
extern const uint8_t nicolas_fp6[];
extern const uint8_t nicolas_output_scales[];
extern const uint8_t output_lut[];
extern const uint8_t output_scales[];
extern const uint8_t source_fp6_packed[];
extern const uint8_t weight[];
extern const uint8_t weight_lut[];
extern const uint8_t weight_scales[];

static uint8_t output_bf16[32768] __attribute__((aligned(64)));
static uint8_t output_quantized[8192] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[512] __attribute__((aligned(64)));

void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_bf16, const void *output_lut, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales);

static uint32_t float_bits(float value) {
  union { float f; uint32_t u; } v = { .f = value };
  return v.u;
}

static float bf16_value(uint16_t bits) {
  union { uint32_t u; float f; } v = { .u = (uint32_t)bits << 16 };
  return v.f;
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

static uint8_t source_fp6_code(uint16_t bits) {
  int sign = (bits & 0x8000) ? -1 : 1;
  int exponent = (bits >> 7) & 255;
  int mantissa = bits & 127;
  float value = 0.0f;
  if (exponent == 255) value = sign * 28.0f;
  else if (exponent != 0) {
    int e = exponent - 127;
    if (e >= -6 && e <= 7) {
      int high = (mantissa >> 5) & 3;
      int rounding = (mantissa >> 4) & 1;
      int sticky = (mantissa & 15) != 0;
      int rounded = high + (rounding & (sticky | (high & 1)));
      int out_e = e + (rounded >= 4);
      value = sign * (out_e > 7 ? 28.0f :
          (1.0f + (rounded >= 4 ? 0 : rounded) * 0.25f) * power_of_two(out_e));
    } else if (e == -7)
      value = sign * (mantissa <= 32 ? 2.0f :
                      mantissa <= 95 ? 3.0f : 4.0f) / 256.0f;
    else if (e == -8)
      value = sign * (mantissa < 64 ? 1.0f : 2.0f) / 256.0f;
    else if (e == -9)
      value = sign * (mantissa == 0 ? 0.0f : 1.0f) / 256.0f;
  }
  float magnitude = value < 0.0f ? -value : value;
  if (magnitude >= 32.0f) value = sign * 28.0f;
  else if (magnitude <= 0.0546875f) value = 0.0f;
  else if (magnitude >= 0.0625f && magnitude <= 0.21875f)
    value = sign * (magnitude <= 0.078125f ? 0.0625f :
                    magnitude <= 0.15625f ? 0.125f : 0.1875f);
  if (value == 0.0f) return 0;
  uint8_t code_sign = value < 0.0f ? 0x20 : 0;
  magnitude = value < 0.0f ? -value : value;
  if (magnitude < 0.25f) {
    int sub = (int)(magnitude / 0.0625f + 0.5f);
    return code_sign | (uint8_t)(sub > 3 ? 3 : sub);
  }
  int out_e = floor_log2_positive_bits(float_bits(magnitude));
  float base = power_of_two(out_e);
  int out_m = (int)((magnitude - base) / (base / 4.0f) + 0.5f);
  if (out_m >= 4) { out_m = 0; ++out_e; }
  int biased = out_e + 3;
  if (biased > 7) biased = 7;
  if (out_m > 3) out_m = 3;
  return code_sign | (uint8_t)((biased << 2) | out_m);
}

static uint8_t lut_code(uint32_t pair, uint32_t index) {
  const uint8_t *line = output_lut + pair * 12;
  uint32_t bit = index * 6, byte = bit >> 3;
  uint16_t chunk = line[byte];
  if (byte + 1 < 12) chunk |= (uint16_t)line[byte + 1] << 8;
  return (uint8_t)((chunk >> (bit & 7)) & 63);
}

static int fp6_fixed(uint8_t code) {
  int exponent = (code >> 2) & 7, mantissa = code & 3;
  if (exponent == 0 && mantissa == 0) return 0;
  int signed_e = exponent == 0 ? -2 : exponent - 3;
  int significand = (exponent == 0 ? 0 : 4) | mantissa;
  int magnitude = (significand << ((signed_e + 2) & 7)) & 255;
  return code & 0x20 ? -magnitude : magnitude;
}

static uint8_t nearest_lut_index(uint32_t pair, uint8_t code) {
  int input = fp6_fixed(code), best_distance = 0;
  uint8_t best = 0;
  for (uint8_t index = 0; index < 16; ++index) {
    int distance = input - fp6_fixed(lut_code(pair, index));
    if (distance < 0) distance = -distance;
    distance &= 0x1ff;
    if (index == 0 || distance < best_distance) {
      best_distance = distance; best = index;
    }
  }
  return best;
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
          floor_log2_positive_bits((uint32_t)maximum << 16) - 4 + 127;
      uint8_t scale_code = maximum == 0 ? 0 :
          (uint8_t)(exponent < 0 ? 0 : exponent > 254 ? 254 : exponent);
      scratch_output_scales[row * 4 + group] = scale_code;
      float scale = power_of_two((int)scale_code - 127);
      for (uint32_t offset = 0; offset < 32; ++offset) {
        uint32_t col = group * 32 + offset;
        float scaled = bf16_value(input[begin + offset]) / scale;
        uint32_t bits = float_bits(scaled);
        uint16_t rounded = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        uint8_t code = source_fp6_code(rounded);
        uint8_t index = nearest_lut_index(row >> 1, code);
        output_quantized[(row >> 1) * 128 + col] |=
            (uint8_t)(index << ((row & 1) ? 4 : 0));
      }
    }
}

int main(void) {
  mx_issue(activation, activation_lut, activation_scales, output_bf16, output_lut, scratch_output_scales, weight, weight_lut, weight_scales);
  radiance_header_requantize();
  int code_errors = 0, scale_errors = 0;
  for (uint32_t i = 0; i < 8192; ++i)
    if (output_quantized[i] != source_fp6_packed[i]) ++code_errors;
  for (uint32_t i = 0; i < 512; ++i)
    if (scratch_output_scales[i] != golden_output_scales[i]) ++scale_errors;
  printf("lowered MX 128x128x128: %d Radiance FP6 packed-index mismatches, %d E8M0 scale mismatches\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}
