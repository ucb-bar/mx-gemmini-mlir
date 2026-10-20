"""Retain target EXP and exact FP8 bytes while removing private FP32 tensors."""

import ctypes
import hashlib
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from test_vpu_rms_packing import EMULATOR

from mx_gemmini_support import gated_projection_packing, vpu_unary_kernel
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)


@pytest.mark.parametrize("width", [32, 160, 5632])
@pytest.mark.parametrize("activation_arithmetic", ["reciprocal", "divide"])
def test_private_projection_values_keep_independent_exp_and_packing_bytes(
    tmp_path, monkeypatch, width, activation_arithmetic
):
    rows, reduction = 16, 32
    with tarfile.open(
        ROOT / "docs/evidence/radiance_rocket_port_20261019/replays.tar.xz", "r:xz"
    ) as archive:
        header = archive.extractfile("replay64/vpu_ref.h").read()
    assert (
        hashlib.sha256(header).hexdigest()
        == "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
    )
    (tmp_path / "vpu_ref.h").write_bytes(header)
    (tmp_path / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())

    def trace(sequence, *, symbol, **unused):
        body = "\n".join(
            f"issue({c.funct},{c.rs1.c_expr()},{c.rs2.c_expr()});"
            for c in sequence
            if isinstance(c, Command)
        )
        return f"static void {symbol}(const void *input,const void *result) {{\n{body}\n}}\n"

    monkeypatch.setattr(vpu_unary_kernel, "emit_c", trace)
    kernel = gated_projection_packing.emit_gated_projection_packing(
        load_profile(PROFILE),
        (rows, width, reduction),
        symbol="packed",
        issuer_symbol="matrix",
        tile_mnk=(16, 16, 32),
        workspace_symbol="workspace",
        activation_arithmetic=activation_arithmetic,
    )
    stub = f"""
unsigned char workspace[{kernel.workspace_bytes + 64}] __attribute__((aligned(64)));
static unsigned calls;
void matrix(const void *a,const void *as,const void *w,const void *ws,const void *out,const void *scratch) {{
 if((const uint8_t *)as!=(const uint8_t *)a+{rows * reduction}u)abort();
 unsigned offset=(calls++%2)*{rows * width * 2}u;
 if(out!=workspace+offset||scratch!=workspace+{rows * width * 4}u)abort();
 memcpy((void *)out,w,{rows * width * 2}u);
}}
void reference(const uint16_t *g,const uint16_t *u,float *out) {{
 for(unsigned i=0;i<{rows * width};++i) {{
  float gate=vpu_bf16_to_f(g[i]);
  float sigmoid=1.0f/(1.0f+vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(-gate))));
  out[i]=(gate*sigmoid)*vpu_bf16_to_f(u[i]);
 }}
}}
"""
    if activation_arithmetic == "divide":
        stub = stub.replace(
            "float sigmoid=1.0f/(1.0f+vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(-gate))));",
            "float denominator=1.0f+vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(-gate)));",
        ).replace(
            "(gate*sigmoid)*vpu_bf16_to_f(u[i])",
            "(gate/denominator)*vpu_bf16_to_f(u[i])",
        )
    source = tmp_path / "packed.c"
    source.write_text(EMULATOR + kernel.source + stub)
    library = tmp_path / "packed.so"
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
    dll.packed.argtypes = [ctypes.c_void_p] * 6
    dll.reference.argtypes = [ctypes.c_void_p] * 3
    dll.mx_pack_activation_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(104)
    gates = rng.normal(0, 4, (rows, width)).astype(np.float32)
    gates[0], gates[1], gates[2], gates[3] = (
        0,
        -np.float32(0),
        np.float32(1e-20),
        np.float32(-90),
    )
    up = rng.normal(size=(rows, width)).astype(np.float32)
    dummy = np.zeros(1, np.uint8)
    input_bytes = np.zeros(kernel.input_packed_bytes, np.uint8)
    workspace = np.ctypeslib.as_array(
        (ctypes.c_uint8 * (kernel.workspace_bytes + 64)).in_dll(dll, "workspace")
    )
    for scale in (1, 0.5):
        gates *= np.float32(scale)
        g = (gates.view(np.uint32) >> 16).astype(np.uint16)
        u = (up.view(np.uint32) >> 16).astype(np.uint16)
        values = np.empty((rows, width), np.float32)
        dll.reference(g.ctypes.data, u.ctypes.data, values.ctypes.data)
        expected = np.empty(kernel.output_packed_bytes, np.uint8)
        assert (
            dll.mx_pack_activation_f32(
                values.ctypes.data,
                rows,
                width,
                8,
                expected.ctypes.data,
                expected[rows * width :].ctypes.data,
            )
            == 0
        )
        output = np.full(kernel.output_packed_bytes + 128, 0xA5, np.uint8)
        workspace[:] = 0xA5
        dll.packed(
            input_bytes.ctypes.data,
            g.ctypes.data,
            dummy.ctypes.data,
            u.ctypes.data,
            dummy.ctypes.data,
            output[64:].ctypes.data,
        )
        assert np.array_equal(output[64:-64], expected)
        assert np.all(output[:64] == 0xA5) and np.all(output[-64:] == 0xA5)
        assert np.all(workspace[-64:] == 0xA5)


@pytest.mark.parametrize(
    "shape", [(15, 32, 32), (16, 48, 32), (16, 8224, 32), (16, 32, 16), (16, 0, 32)]
)
def test_incomplete_or_oversized_rows_are_refused(shape):
    with pytest.raises(ValueError):
        gated_projection_packing.emit_gated_projection_packing(
            load_profile(PROFILE),
            shape,
            symbol="packed",
            issuer_symbol="matrix",
            tile_mnk=(16, 16, 32),
        )
