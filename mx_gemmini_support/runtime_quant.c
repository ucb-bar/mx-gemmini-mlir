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

#if !defined(MX_QUANT_LUT_GENERATOR) && !defined(MX_QUANT_REFERENCE)
#include "runtime_quant_lut.h"

/* Power-of-two scaling preserves a rounded BF16 mantissa. The scale exponent
 * and that mantissa therefore select the same code as the reference search. */
#if defined(MX_QUANT_GROUP_WORDS)
__attribute__((unused))
#endif
static uint8_t fast_element_code(float rounded, int scale_exponent,
                                 uint32_t format) {
  uint32_t bits = bits_of_float(rounded);
  uint32_t sign = bits >> 31;
  uint32_t biased_exponent = (bits >> 23) & 255u;
  if (biased_exponent == 0u)
    return format == MX_RUNTIME_FP8 ? (uint8_t)(sign << 7) : 0u;
  int delta = scale_exponent + 127 - (int)biased_exponent;
  if (delta < 0)
    return element_code(rounded / power_of_two(scale_exponent), format);
  if (delta >= 16)
    return format == MX_RUNTIME_FP8 ? (uint8_t)(sign << 7) : 0u;
  uint32_t mantissa = (bits >> 16) & 127u;
  uint8_t code = (format == MX_RUNTIME_FP4
                  ? mx_fp4_code_lut[delta][mantissa]
                  : mx_fp8_code_lut[delta][mantissa]);
  if (format == MX_RUNTIME_FP4)
    return code ? (uint8_t)(code | (sign << 3)) : 0u;
  return (uint8_t)(code | (sign << 7));
}
#endif

static int block_scale_exponent(float maximum) {
  if (maximum < 0x1p-23f) maximum = 0x1p-23f;
  int exponent = (int)((bits_of_float(maximum) >> 23) & 255u) - 127;
  if (exponent < -23) exponent = -23;
  if (exponent > 127) exponent = 127;
  return exponent;
}

#if defined(MX_QUANT_GROUP_WORDS) && !defined(MX_QUANT_REFERENCE) && !defined(MX_QUANT_LUT_GENERATOR)
static inline int mx_round_group_word(float value, uint16_t *word) {
  *word = (uint16_t)(bits_of_float(rounded_bf16(value)) >> 16);
  return (*word & 0x7f80u) == 0x7f80u ? -1 : 0;
}
static inline uint16_t mx_group_word_magnitude(uint16_t word) { return word & 0x7fffu; }
#if defined(MX_QUANT_BF16_INPUT)
static inline int mx_copy_group_word(uint16_t value, uint16_t *word) {
  *word = value;
  return (value & 0x7f80u) == 0x7f80u ? -1 : 0;
}
#endif
#include "runtime_group_prepare.h"

/* The caller has prepared all 32 finite words and their unsigned magnitude
 * maximum. Keep target scale selection, codebook and nibble layout shared by
 * float and word input routes. Inlining retains each caller's fixed axis.
 */
static inline __attribute__((always_inline))
void pack_prepared_group(const uint16_t *values, uint16_t maximum,
                         uint32_t rows, uint32_t cols, uint32_t axis,
                         uint32_t major, uint32_t group, uint32_t format,
                         uint8_t *codes, uint8_t *scales) {
  int exponent = (int)(maximum >> 7) - 127;
  if (exponent < -23) exponent = -23;
  uint32_t group_major = axis == 1 ? rows : cols;
  scales[(size_t)group * group_major + major] = (uint8_t)(exponent + 127);
  /* Select the codebook and destination layout once per group. Keeping the
   * FP8 path separate avoids FP4 nibble and format branches for every member.
   * The prepared words, scale, traversal and signed-zero encoding are unchanged.
   */
  if (format == MX_RUNTIME_FP8) {
    size_t start = axis == 1 ? (size_t)major * cols + group * 32u
                             : (size_t)group * 32u * cols + major;
    size_t stride = axis == 1 ? 1u : cols;
    uint32_t scale_biased = (uint32_t)(exponent + 127);
    for (uint32_t offset = 0; offset < 32; ++offset) {
      uint32_t word = values[offset];
      uint32_t biased = (word >> 7) & 255u;
      uint32_t delta = scale_biased - biased;
      uint8_t code = (uint8_t)((word >> 15) << 7);
      if (biased != 0u && delta < 16u)
        code |= mx_fp8_code_lut[delta][word & 127u];
      codes[start + offset * stride] = code;
    }
    return;
  }
  for (uint32_t offset = 0; offset < 32; ++offset) {
    uint32_t row = axis == 1 ? major : group * 32u + offset;
    uint32_t col = axis == 1 ? group * 32u + offset : major;
    uint32_t word = values[offset];
    uint32_t biased = (word >> 7) & 255u;
    uint8_t code;
    /* Every finite member's exponent is at most the group's maximum. The
     * lower scale clamp can only increase the selected exponent, so delta is
     * nonnegative. BF16 magnitudes have the same order as their unsigned words.
     * Subnormals are below the minimum scale's representable code threshold.
     */
    uint32_t delta = (uint32_t)(exponent + 127) - biased;
    if (biased == 0u || delta >= 16u) {
      code = format == MX_RUNTIME_FP8 ? (uint8_t)((word >> 15) << 7) : 0u;
    } else {
      uint32_t mantissa = word & 127u;
      code = format == MX_RUNTIME_FP4 ? mx_fp4_code_lut[delta][mantissa] : mx_fp8_code_lut[delta][mantissa];
      if (format == MX_RUNTIME_FP8) code |= (uint8_t)((word >> 15) << 7);
      else if (code) code |= (uint8_t)((word >> 15) << 3);
    }
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
}
#endif

#if defined(MX_QUANT_INLINE_GROUP)
static inline __attribute__((always_inline))
#else
static
#endif
int pack_one_group(const float *input, uint32_t rows, uint32_t cols,
                          uint32_t axis, uint32_t major, uint32_t group,
                          uint32_t format, uint8_t *codes, uint8_t *scales) {
#if defined(MX_QUANT_GROUP_WORDS) && !defined(MX_QUANT_REFERENCE) && !defined(MX_QUANT_LUT_GENERATOR)
  uint16_t values[32];
  uint16_t maximum = 0;
  const float *start = input + (axis == 1 ? (size_t)major * cols + group * 32u :
                                         (size_t)group * 32u * cols + major);
  if (mx_prepare_group_words(start, axis == 1 ? 1u : cols, values, &maximum)) return -1;
  pack_prepared_group(values, maximum, rows, cols, axis, major, group, format, codes, scales);
#else
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
  int exponent = block_scale_exponent(maximum);
  uint32_t group_major = axis == 1 ? rows : cols;
  scales[(size_t)group * group_major + major] = (uint8_t)(exponent + 127);
#if defined(MX_QUANT_LUT_GENERATOR) || defined(MX_QUANT_REFERENCE)
  float scale = power_of_two(exponent);
#endif
  for (uint32_t offset = 0; offset < 32; ++offset) {
    uint32_t row = axis == 1 ? major : group * 32u + offset;
    uint32_t col = axis == 1 ? group * 32u + offset : major;
#if defined(MX_QUANT_LUT_GENERATOR) || defined(MX_QUANT_REFERENCE)
    uint8_t code = element_code(values[offset] / scale, format);
#else
    uint8_t code = fast_element_code(values[offset], exponent, format);
#endif
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
#endif
  return 0;
}

static int valid_shape(const void *input, uint32_t rows, uint32_t cols,
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

int mx_pack_activation_zero_rows_f32(const float *input, uint32_t live_rows,
                                    uint32_t m, uint32_t k, uint32_t format,
                                    uint8_t *codes, uint8_t *scales) {
  if (!valid_shape(input, m, k, 1, format, codes, scales) ||
      !live_rows || live_rows > m) return -1;
  uint8_t zero_scale = (uint8_t)(block_scale_exponent(0.0f) + 127);
  for (uint32_t group = 0; group < k / 32u; ++group) {
    for (uint32_t row = 0; row < m; ++row) {
      if (row < live_rows) {
        if (pack_one_group(input, m, k, 1, row, group, format, codes, scales)) return -1;
      } else {
        scales[(size_t)group * m + row] = zero_scale;
        /* An FP4 even row writes the whole byte and clears the high nibble;
         * the following zero odd row therefore needs no code writes.
         */
        if (format == MX_RUNTIME_FP8 || !(row & 1u)) {
          size_t start = (size_t)(format == MX_RUNTIME_FP8 ? row : row / 2u) * k + group * 32u;
          for (uint32_t col = 0; col < 32u; ++col) codes[start + col] = 0;
        }
      }
    }
  }
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

#if defined(MX_QUANT_BF16_INPUT)
#if !defined(MX_QUANT_GROUP_WORDS) || defined(MX_QUANT_REFERENCE) || defined(MX_QUANT_LUT_GENERATOR)
#error "BF16 input needs the selected word-group LUT preparation"
#endif
int mx_pack_activation_bf16(const uint16_t *input, uint32_t m, uint32_t k,
                            uint32_t format, uint8_t *codes, uint8_t *scales) {
  /* valid_shape retains the original conservative f32 byte-span bound. */
  if (!valid_shape(input, m, k, 1, format, codes, scales)) return -1;
  for (uint32_t group = 0; group < k / 32u; ++group) {
    for (uint32_t row = 0; row < m; ++row) {
      uint16_t values[32], maximum;
      if (mx_prepare_bf16_group_words(input + (size_t)row * k + group * 32u,
                                      1u, values, &maximum)) return -1;
      pack_prepared_group(values, maximum, m, k, 1u, row, group, format, codes, scales);
    }
  }
  return 0;
}

int mx_pack_activation_bf16_rows(const uint16_t *input, uint32_t m, uint32_t k,
                                uint32_t first, uint32_t count,
                                uint8_t *codes, uint8_t *scales) {
  if (!valid_shape(input, m, k, 1, MX_RUNTIME_FP8, codes, scales) ||
      first >= m || !count || count > m - first) return -1;
  for (uint32_t group = 0; group < k / 32u; ++group) {
    for (uint32_t row = 0; row < count; ++row) {
      uint16_t values[32], maximum;
      if (mx_prepare_bf16_group_words(input + (size_t)row * k + group * 32u,
                                      1u, values, &maximum)) return -1;
      pack_prepared_group(values, maximum, m, k, 1u, first + row, group,
                          MX_RUNTIME_FP8, codes, scales);
    }
  }
  return 0;
}
#endif
