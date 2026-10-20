"""Consume two immutable FP8 projections as BF16 before gated FP32 packing."""

from dataclasses import dataclass

from .fp8_row_packing import emit_fp8_row_packer
from .scheduled_native_issuer import emit_scheduled_native_issuer
from .vpu_unary_kernel import _issue


@dataclass(frozen=True)
class GatedProjectionPackingKernel:
    source: str
    workspace_bytes: int
    input_packed_bytes: int
    output_packed_bytes: int
    schedule_digest: str


def emit_gated_projection_packing(
    profile,
    shape,
    *,
    symbol,
    issuer_symbol,
    tile_mnk,
    workspace_symbol="mx_catalog_workspace",
    activation_arithmetic="reciprocal",
):
    """Retain both original matrix schedules and the hybrid sigmoid arithmetic.

    All intermediates are private. Original matrix issuers complete before
    vector work. Each row evaluates (gate * (1 / (1 + exp(-gate)))) * up,
    with BF16 EXP and FP32 arithmetic, then canonical BF16-input FP8 packing.
    Explicit divide arithmetic evaluates (gate / (1 + exp(-gate))) * up
    instead; this changes rounding and requires separate numerical acceptance.
    The caller proves static shapes, constant weights and complete use closure.
    """
    if activation_arithmetic not in {"reciprocal", "divide"}:
        raise ValueError("unknown gated activation arithmetic")
    if (
        len(shape) != 3
        or any(type(d) is not int or d <= 0 for d in shape)
        or any(
            not s.isascii() or not s.isidentifier()
            for s in (symbol, issuer_symbol, workspace_symbol)
        )
    ):
        raise ValueError(
            "gated projection packing requires static extents and C symbols"
        )
    if (
        profile.get("transport") != "rocket_rocc"
        or profile.get("geometry", {}).get("mesh_columns") != 16
        or not profile.get("resources", {}).get("vpu")
        or profile["resources"].get("scale_block_elements") != 32
    ):
        raise ValueError(
            "gated projection packing requires the DIM16 Rocket block32 VPU"
        )
    m, n, k = shape
    if (
        m % 16
        or n % 32
        or n > 8192
        or k % 32
        or m * n > 0x7FFFFFFF
        or m * k > 0x7FFFFFFF
    ):
        raise ValueError(
            "gated projection packing requires complete FP8 matrices and one VPU row tile"
        )
    native = emit_scheduled_native_issuer(
        profile, shape, precision="FP8", symbol=issuer_symbol, tile_mnk=tuple(tile_mnk)
    )
    if dict(native.issuer.buffer_bytes) != {
        "activation": m * k,
        "activation_scales": m * k // 32,
        "weight": k * n,
        "weight_scales": k * n // 32,
        "output_bf16": m * n * 2,
        "scratch_output_scales": 2048,
    }:
        raise ValueError(
            "gated projection buffer extents differ from the matrix schedule"
        )
    align = lambda size: (size + 63) // 64 * 64
    gate_offset = 0
    up_offset = align(m * n * 2)
    scratch = align(up_offset + m * n * 2)
    negative = align(scratch + 2048)
    exponential = align(negative + n * 2)
    values = align(exponential + n * 2)
    workspace = align(values + n * 4)
    if workspace > 0x7FFFFFFF:
        raise ValueError("gated projection workspace exceeds the runtime extent")
    source = (
        "#include <stdint.h>\n"
        + _issue(profile, elements=n, operation="exp", symbol=symbol + "_exp")
        + emit_fp8_row_packer(rows=m, width=n, symbol=symbol + "_pack")
    )
    arithmetic = (
        "float sigmoid=1.0f/(1.0f+e.f);\n      values[col]=(g.f*sigmoid)*u.f;"
        if activation_arithmetic == "reciprocal"
        else "values[col]=(g.f/(1.0f+e.f))*u.f;"
    )
    source += f"""
extern unsigned char {workspace_symbol}[];
void {issuer_symbol}(const void *,const void *,const void *,const void *,const void *,const void *);
void {symbol}(const uint8_t *input,const uint8_t *gate_codes,const uint8_t *gate_scales,
              const uint8_t *up_codes,const uint8_t *up_scales,uint8_t *output) {{
  if (!input || !gate_codes || !gate_scales || !up_codes || !up_scales || !output) __builtin_trap();
  {issuer_symbol}(input,input+{m * k}u,gate_codes,gate_scales,
                 {workspace_symbol}+{gate_offset}u,{workspace_symbol}+{scratch}u);
  {issuer_symbol}(input,input+{m * k}u,up_codes,up_scales,
                 {workspace_symbol}+{up_offset}u,{workspace_symbol}+{scratch}u);
  const uint16_t *gate=(const uint16_t *)({workspace_symbol}+{gate_offset}u);
  const uint16_t *up=(const uint16_t *)({workspace_symbol}+{up_offset}u);
  uint16_t *negative=(uint16_t *)({workspace_symbol}+{negative}u);
  uint16_t *exponential=(uint16_t *)({workspace_symbol}+{exponential}u);
  float *values=(float *)({workspace_symbol}+{values}u);
  for (uint32_t row=0;row<{m}u;++row) {{
    for (uint32_t col=0;col<{n}u;++col) {{
      uint16_t g=gate[row*{n}u+col];
      if ((g&0x7f80u)==0x7f80u) __builtin_trap();
      negative[col]=g^0x8000u;
    }}
    {symbol}_exp(negative,exponential);
    for (uint32_t col=0;col<{n}u;++col) {{
      union {{uint32_t u;float f;}} g={{.u=(uint32_t)gate[row*{n}u+col]<<16}},
        u={{.u=(uint32_t)up[row*{n}u+col]<<16}},e={{.u=(uint32_t)exponential[col]<<16}};
      {arithmetic}
    }}
    if ({symbol}_pack(values,row,1,output,output+{m * n}u)) __builtin_trap();
  }}
}}
"""
    return GatedProjectionPackingKernel(
        source, workspace, m * k * 33 // 32, m * n * 33 // 32, native.digest
    )
