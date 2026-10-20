"""Specialize the canonical BF16-input FP8 group layout for private row tiles."""


def emit_fp8_row_packer(*, rows, width, symbol):
    """Emit fixed-stride packing; the caller publishes only complete tensors.

    Codes are row major; E8M0 scales are group major. Input FP32 values round
    to BF16 with ties to even before selecting each block32 scale, exactly as
    the runtime quantizer. Nonfinite words refuse before that group's writes.
    """
    if (
        type(rows) is not int
        or rows < 16
        or rows % 16
        or type(width) is not int
        or width < 32
        or width % 32
        or rows * width > 0x7FFFFFFF
        or not symbol.isascii()
        or not symbol.isidentifier()
    ):
        raise ValueError("FP8 row packing requires complete static matrix extents")
    from .runtime_group_preparation import emit_runtime_group_preparation

    prepare = (
        emit_runtime_group_preparation()
        .replace("mx_prepare_group_words", symbol + "_prepare")
        .replace("mx_round_group_word", symbol + "_round")
        .replace("mx_group_word_magnitude", symbol + "_magnitude")
    )
    return f"""#include <stdint.h>
#include "runtime_quant_lut.h"
static inline int {symbol}_round(float input,uint16_t *value) {{
  union {{float f;uint32_t u;}} word={{.f=input}};
  word.u+=0x7fffu+((word.u>>16)&1u);
  *value=word.u>>16;
  return (*value&0x7f80u)==0x7f80u ? -1 : 0;
}}
static inline uint16_t {symbol}_magnitude(uint16_t word) {{ return word&0x7fffu; }}
{prepare}
static inline int {symbol}(const float *input,uint32_t first,uint32_t count,
                           uint8_t *codes,uint8_t *scales) {{
  if (!input || !codes || !scales || first>={rows}u || !count || count>{rows}u-first) return -1;
  for (uint32_t group=0; group<{width // 32}u; ++group)
    for (uint32_t row=0; row<count; ++row) {{
      uint16_t values[32],maximum;
      if ({symbol}_prepare(input+row*{width}u+group*32u,1u,values,&maximum)) return -1;
      int exponent=(int)(maximum>>7)-127;
      if (exponent < -23) exponent=-23;
      uint32_t biased_scale=exponent+127;
      scales[group*{rows}u+first+row]=(uint8_t)biased_scale;
      for (uint32_t j=0; j<32; ++j) {{
        uint32_t value=values[j],biased=(value>>7)&255u,delta=biased_scale-biased;
        uint8_t code=(value>>15)<<7;
        if (biased && delta<16u) code|=mx_fp8_code_lut[delta][value&127u];
        codes[(first+row)*{width}u+group*32u+j]=code;
      }}
    }}
  return 0;
}}
"""
