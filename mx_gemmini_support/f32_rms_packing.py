"""Pack private FP32 RMS outputs without materializing normalization tensors."""

from dataclasses import dataclass

from .fp8_row_packing import emit_fp8_row_packer


@dataclass(frozen=True)
class F32RMSPackingKernel:
    source: str
    workspace_bytes: int
    packed_bytes: int


def emit_f32_rms_packing(
    profile,
    *,
    rows,
    width,
    symbol,
    reduction="ordered",
    workspace_symbol="mx_catalog_workspace",
):
    if (
        type(rows) is not int
        or type(width) is not int
        or rows <= 0
        or rows % 16
        or not 32 <= width <= 8192
        or width % 32
        or rows * width * 4 > 0x7FFFFFFF
        or reduction not in {"ordered", "striped16"}
        or reduction == "striped16"
        and width % 256
        or any(
            not x.isascii() or not x.isidentifier() for x in (symbol, workspace_symbol)
        )
    ):
        raise ValueError(
            "FP32 RMS packing needs bounded complete rows and a supported reduction"
        )
    if (
        profile.get("transport") != "rocket_rocc"
        or profile.get("geometry", {}).get("mesh_columns") != 16
        or profile.get("resources", {}).get("scale_block_elements") != 32
    ):
        raise ValueError("FP32 RMS packing requires the Rocket DIM16 block32 target")
    if reduction == "ordered":
        summation = f"""float sum=0.0f;
    for(uint32_t col=0;col<{width}u;++col){{float x=input[row*{width}u+col];sum=sum+x*x;}}"""
    else:
        summation = f"""float partial[16]={{0}};
    for(uint32_t lane=0;lane<16u;++lane)
      for(uint32_t word=lane;word<{width // 2}u;word+=128u)
        for(uint32_t j=0;j<8u;++j){{
          uint32_t col=2u*(word+j*16u);
          float p=input[row*{width}u+col],q=input[row*{width}u+col+1u];
          partial[lane]=partial[lane]+(p*p+q*q);
        }}
    float sum=0.0f;
    for(uint32_t lane=0;lane<16u;++lane)sum=sum+partial[lane];"""
    source = "#include <stdint.h>\n" + emit_fp8_row_packer(
        rows=rows, width=width, symbol=symbol + "_pack"
    )
    source += f"""
extern unsigned char {workspace_symbol}[];
void {symbol}(const float *input,const float *gamma,const float *divisor,const float *epsilon,uint8_t *output){{
  if(!input||!gamma||!divisor||!epsilon||!output||*divisor!={float(width)}f)__builtin_trap();
  float *values=(float *){workspace_symbol};
  for(uint32_t row=0;row<{rows}u;++row){{
    {summation}
    float inverse=1.0f/__builtin_sqrtf(sum/(*divisor)+*epsilon);
    for(uint32_t col=0;col<{width}u;++col)values[col]=gamma[col]*(input[row*{width}u+col]*inverse);
    if({symbol}_pack(values,row,1,output,output+{rows * width}u))__builtin_trap();
  }}
}}
"""
    return F32RMSPackingKernel(
        source, (width * 4 + 63) // 64 * 64, rows * width * 33 // 32
    )
