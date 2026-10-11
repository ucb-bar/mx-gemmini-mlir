/* Target-owned live operand conversion for the MX packed-pointer ABI.
 * Numerical policy: MXQuant bf16_e8m0_block32_rne, then target packing.
 * No heap, PyTorch, Python or embedded tensor payloads are used at runtime.
 */
#include "runtime_quant.h"

#include <stddef.h>

static float float_of_bits(uint32_t bits) {
  union { uint32_t bits; float value; } word;
  word.bits = bits;
  return word.value;
}

static uint32_t bits_of_float(float value) {
  union { uint32_t bits; float value; } word;
  word.value = value;
  return word.bits;
}

static float rounded_bf16(float value) {
  uint32_t bits = bits_of_float(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return float_of_bits(bits & 0xffff0000u);
}

static float power_of_two(int exponent) {
  /* Every exponent used below is a normal finite binary32 value. */
  return float_of_bits((uint32_t)(exponent + 127) << 23);
}

static float positive_grid(uint32_t code, uint32_t exponent_bits,
                           uint32_t fraction_bits, int bias) {
  (void)exponent_bits;
  uint32_t exponent = code >> fraction_bits;
  uint32_t fraction = code & ((1u << fraction_bits) - 1u);
  if (exponent == 0)
    return (float)fraction * power_of_two(1 - bias - (int)fraction_bits);
  return (1.0f + (float)fraction / (float)(1u << fraction_bits)) *
         power_of_two((int)exponent - bias);
}

static uint32_t rounded_positive_code(float magnitude, uint32_t exp_bits,
                                      uint32_t frac_bits, int bias,
                                      uint32_t max_code) {
  uint32_t low = 0, high = max_code;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2u;
    if (positive_grid(mid, exp_bits, frac_bits, bias) < magnitude)
      low = mid + 1u;
    else
      high = mid;
  }
  uint32_t upper = low;
  uint32_t lower = upper == 0 ? 0 : upper - 1u;
  float lower_gap = magnitude - positive_grid(lower, exp_bits, frac_bits, bias);
  float upper_gap = positive_grid(upper, exp_bits, frac_bits, bias) - magnitude;
  return (upper_gap < lower_gap || (upper_gap == lower_gap && (lower & 1u)))
             ? upper : lower;
}

static uint8_t element_code(float normalized, uint32_t format) {
  uint32_t bits = bits_of_float(normalized);
  uint32_t negative = bits >> 31;
  float magnitude = float_of_bits(bits & 0x7fffffffu);
  if (format == MX_RUNTIME_FP4) {
    uint32_t intermediate = rounded_positive_code(magnitude, 3, 1, 3, 15);
    magnitude = positive_grid(intermediate, 3, 1, 3);
    uint32_t code = rounded_positive_code(magnitude, 2, 1, 1, 7);
    return code == 0 ? 0 : (uint8_t)(code | (negative << 3));
  }
  return (uint8_t)(rounded_positive_code(magnitude, 4, 3, 7, 0x7e) |
                   (negative << 7));
}

static int pack_one_group(const float *input, uint32_t rows, uint32_t cols,
                          uint32_t axis, uint32_t major, uint32_t group,
                          uint32_t format, uint8_t *codes, uint8_t *scales) {
  float values[32];
  float maximum = 0.0f;
  for (uint32_t offset = 0; offset < 32; ++offset) {
    uint32_t row = axis == 1 ? major : group * 32u + offset;
    uint32_t col = axis == 1 ? group * 32u + offset : major;
    float value = rounded_bf16(input[(size_t)row * cols + col]);
    if (((bits_of_float(value) >> 23) & 255u) == 255u) return -1;
    values[offset] = value;
    float magnitude = float_of_bits(bits_of_float(value) & 0x7fffffffu);
    if (magnitude > maximum) maximum = magnitude;
  }
  if (maximum < 0x1p-23f) maximum = 0x1p-23f;
  int exponent = (int)((bits_of_float(maximum) >> 23) & 255u) - 127;
  if (exponent < -23) exponent = -23;
  if (exponent > 127) exponent = 127;
  uint32_t group_major = axis == 1 ? rows : cols;
  scales[(size_t)group * group_major + major] = (uint8_t)(exponent + 127);
  float scale = power_of_two(exponent);
  for (uint32_t offset = 0; offset < 32; ++offset) {
    uint32_t row = axis == 1 ? major : group * 32u + offset;
    uint32_t col = axis == 1 ? group * 32u + offset : major;
    uint8_t code = element_code(values[offset] / scale, format);
    if (format == MX_RUNTIME_FP8) {
      codes[(size_t)row * cols + col] = code;
    } else if (axis == 1) {
      size_t index = (size_t)(row / 2u) * cols + col;
      if (row & 1u) codes[index] |= (uint8_t)(code << 4);
      else codes[index] = code;
    } else {
      size_t index = (size_t)row * (cols / 2u) + col / 2u;
      if (col & 1u) codes[index] |= (uint8_t)(code << 4);
      else codes[index] = code;
    }
  }
  return 0;
}

static int valid_shape(const float *input, uint32_t rows, uint32_t cols,
                       uint32_t axis, uint32_t format,
                       const uint8_t *codes, const uint8_t *scales) {
  uint32_t tile = format == MX_RUNTIME_FP8 ? 16u : 32u;
  return input && codes && scales &&
         (format == MX_RUNTIME_FP8 || format == MX_RUNTIME_FP4) &&
         rows >= tile && cols >= tile && rows % tile == 0 && cols % tile == 0 &&
         (axis == 1 ? cols % 32u == 0 : rows % 32u == 0) &&
         (size_t)rows * cols <= SIZE_MAX / sizeof(float);
}

int mx_pack_activation_f32(const float *input, uint32_t m, uint32_t k,
                           uint32_t format, uint8_t *codes, uint8_t *scales) {
  if (!valid_shape(input, m, k, 1, format, codes, scales)) return -1;
  for (uint32_t group = 0; group < k / 32u; ++group)
    for (uint32_t row = 0; row < m; ++row)
      if (pack_one_group(input, m, k, 1, row, group, format, codes, scales)) return -1;
  return 0;
}

int mx_pack_weight_f32(const float *input, uint32_t k, uint32_t n,
                       uint32_t format, uint8_t *codes, uint8_t *scales) {
  if (!valid_shape(input, k, n, 0, format, codes, scales)) return -1;
  for (uint32_t group = 0; group < k / 32u; ++group)
    for (uint32_t col = 0; col < n; ++col)
      if (pack_one_group(input, k, n, 0, col, group, format, codes, scales)) return -1;
  return 0;
}

int mx_widen_bf16_f32(const uint16_t *input, uint32_t count, float *output) {
  if (!input || !output) return -1;
  for (uint32_t index = 0; index < count; ++index)
    output[index] = float_of_bits((uint32_t)input[index] << 16);
  return 0;
}
