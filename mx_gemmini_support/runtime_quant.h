#ifndef MX_GEMMINI_RUNTIME_QUANT_H
#define MX_GEMMINI_RUNTIME_QUANT_H

#include <stdint.h>

/* FP8 E4M3 or FP4 E2M1 through the MX E3M1 intermediate. */
#define MX_RUNTIME_FP8 8u
#define MX_RUNTIME_FP4 4u

/* Return zero on success. Dimensions must already be padded to the MX tile. */
int mx_pack_activation_f32(const float *input, uint32_t m, uint32_t k,
                           uint32_t format, uint8_t *codes, uint8_t *scales);
int mx_pack_weight_f32(const float *input, uint32_t k, uint32_t n,
                       uint32_t format, uint8_t *codes, uint8_t *scales);

/* Widen one BF16 readout tensor into its model-visible f32 buffer. */
int mx_widen_bf16_f32(const uint16_t *input, uint32_t count, float *output);

#endif
