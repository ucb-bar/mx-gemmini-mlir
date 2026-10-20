"""Compare streamed RMS/packing with the independent target arithmetic reference."""

import ctypes
import hashlib
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support import vpu_rms_packing
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)

EMULATOR = r"""
#include <stdlib.h>
#include "vpu_ref.h"
static uint16_t scratch[16384][8];
static void issue(unsigned op,uint64_t a,uint64_t b) {
 if(op==0||op==7)return;
 if(op==2||op==3){
  unsigned start=(unsigned)b,rows=b>>48,bytes=(b>>32)&0xffff;
  if(bytes!=16||start+rows>16384)abort();
  if(op==2)memcpy(scratch[start],(const void *)(uintptr_t)a,rows*16);
  else memcpy((void *)(uintptr_t)a,scratch[start],rows*16);
  return;
 }
 if(op!=33)abort();
 unsigned code=b&15,src=a&0x3fff,dst=(a>>28)&0x3fff,other=(a>>14)&0x3fff;
 unsigned rows=(a>>42)&0xffff,rlen=(b>>5)&1023,bcast=(b>>4)&1;
 unsigned result_rows=vpu_is_reduction(code)?rows/rlen:rows;
 if(src+rows>16384||dst+result_rows>16384)abort();
 if(vpu_uses_src2(code)&&other+(bcast?(rows+rlen-1)/rlen:rows)>16384)abort();
 vpu_ref_exec(code,scratch[dst],scratch[src],scratch[other],rows,rlen,bcast,b>>16);
}
"""


@pytest.mark.parametrize("rows,width", [(16, 32), (48, 256), (16, 2048)])
def test_streamed_packed_bytes_equal_original_bf16_sequence(
    tmp_path, monkeypatch, rows, width
):
    with tarfile.open(
        ROOT / "docs/evidence/radiance_rocket_port_20261019/replays.tar.xz", "r:xz"
    ) as archive:
        header = archive.extractfile("replay64/vpu_ref.h").read()
    assert (
        hashlib.sha256(header).hexdigest()
        == "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
    )
    (tmp_path / "vpu_ref.h").write_bytes(header)
    (tmp_path / "runtime_group_prepare.h").write_text(
        emit_runtime_group_preparation(bf16_input=True)
    )

    def trace(sequence, *, symbol, **unused):
        body = "\n".join(
            f"issue({c.funct},{c.rs1.c_expr()},{c.rs2.c_expr()});"
            for c in sequence
            if isinstance(c, Command)
        )
        return f"static void {symbol}(const void *x,const void *gamma,const void *epsilon,const void *result) {{\n{body}\n}}\n"

    monkeypatch.setattr(vpu_rms_packing, "emit_c", trace)
    kernel = vpu_rms_packing.emit_vpu_rms_packing(
        load_profile(PROFILE), rows=rows, width=width, symbol="streamed"
    )
    reference = f"""
void reference(const float *x,const float *g,const float *d,const float *e,uint16_t *out) {{
 for(unsigned row=0;row<{rows};row++) {{
  uint16_t square[{width // 8}][8],sum[1][8];
  for(unsigned col=0;col<{width};col++){{uint16_t a=vpu_f_to_bf16(x[row*{width}+col]);square[col/8][col%8]=vpu_mul(a,a);}}
  vpu_ref_exec(VPU_RSUM,sum,square,0,{width // 8},{width // 8},0,0);
  uint16_t mean=vpu_mul(sum[0][0],vpu_rcp(vpu_f_to_bf16(*d)));
  uint16_t inv=vpu_rsqrt(vpu_add(mean,vpu_f_to_bf16(*e)));
  for(unsigned col=0;col<{width};col++)out[row*{width}+col]=vpu_mul(vpu_mul(vpu_f_to_bf16(x[row*{width}+col]),inv),vpu_f_to_bf16(g[col]));
 }}
}}
"""
    source = tmp_path / "rms.c"
    source.write_text(EMULATOR + kernel.source + reference)
    library = tmp_path / "rms.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DMX_QUANT_GROUP_WORDS",
            "-DMX_QUANT_INLINE_GROUP",
            "-DMX_QUANT_BF16_INPUT",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "mx_gemmini_support"),
            str(source),
            str(ROOT / "mx_gemmini_support/runtime_quant.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    for name in ("streamed", "reference"):
        getattr(dll, name).argtypes = [ctypes.c_void_p] * 5
    dll.mx_pack_activation_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    dll.mx_pack_activation_bf16_rows.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
    rng = np.random.default_rng(12)
    values = rng.normal(size=(rows, width)).astype(np.float32)
    values[0] = 0
    values[1] = -np.float32(0)
    values[2] *= 1e-20
    values[3] *= 1e5
    gamma = rng.uniform(0.5, 1.5, width).astype(np.float32)
    divisor, epsilon = np.array(width, np.float32), np.array(1e-5, np.float32)
    expected_words = np.empty(values.shape, np.uint16)
    dll.reference(
        values.ctypes.data,
        gamma.ctypes.data,
        divisor.ctypes.data,
        epsilon.ctypes.data,
        expected_words.ctypes.data,
    )
    expected = np.empty(kernel.packed_bytes, np.uint8)
    assert (
        dll.mx_pack_activation_bf16(
            expected_words.ctypes.data,
            rows,
            width,
            8,
            expected.ctypes.data,
            expected[rows * width :].ctypes.data,
        )
        == 0
    )
    for scale in (1, 0.5):
        # Repeated calls must refresh input data and private scratchpad state.
        values *= np.float32(scale)
        dll.reference(
            values.ctypes.data,
            gamma.ctypes.data,
            divisor.ctypes.data,
            epsilon.ctypes.data,
            expected_words.ctypes.data,
        )
        assert (
            dll.mx_pack_activation_bf16(
                expected_words.ctypes.data,
                rows,
                width,
                8,
                expected.ctypes.data,
                expected[rows * width :].ctypes.data,
            )
            == 0
        )
        output = np.full(kernel.packed_bytes + 2, 0xA5, np.uint8)
        dll.streamed(
            values.ctypes.data,
            gamma.ctypes.data,
            divisor.ctypes.data,
            epsilon.ctypes.data,
            output[1:].ctypes.data,
        )
        assert np.array_equal(output[1:-1], expected)
        assert output[0] == output[-1] == 0xA5
    for first, count in ((rows, 1), (0, 0), (rows - 1, 2)):
        untouched = np.full(kernel.packed_bytes, 0xA5, np.uint8)
        assert dll.mx_pack_activation_bf16_rows(
            expected_words.ctypes.data, rows, width, first, count,
            untouched.ctypes.data, untouched[rows * width:].ctypes.data) == -1
        assert np.all(untouched == 0xA5)
