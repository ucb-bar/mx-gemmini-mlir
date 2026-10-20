"""Hybrid sigmoid uses the pinned VPU exponential and an explicit FP32 reciprocal."""

import ctypes
import hashlib
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from test_vpu_rms_packing import EMULATOR

from mx_gemmini_support import vpu_unary_kernel
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)


@pytest.mark.parametrize("elements", [7, 8193, 160 * 5632])
def test_every_output_word_matches_independent_target_exp_reference(
    tmp_path, monkeypatch, elements
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
    commands = []

    def trace(sequence, *, symbol, **unused):
        commands.extend(sequence)
        body = "\n".join(
            f"issue({c.funct},{c.rs1.c_expr()},{c.rs2.c_expr()});"
            for c in sequence
            if isinstance(c, Command)
        )
        return f"static void {symbol}(const void *input,const void *result) {{\n{body}\n}}\n"

    monkeypatch.setattr(vpu_unary_kernel, "emit_c", trace)
    kernel = vpu_unary_kernel.emit_vpu_unary(
        load_profile(PROFILE),
        shape=(elements,),
        dtype="f32",
        operation="sigmoid",
        symbol="hybrid",
        sigmoid_implementation="f32_reciprocal",
    )
    reference = f"""void reference(const float *x,float *y) {{
      for(unsigned i=0;i<{elements}u;i++){{
        uint16_t exponential=vpu_exp(vpu_f_to_bf16(-x[i]));
        y[i]=1.0f/(1.0f+vpu_bf16_to_f(exponential));
      }}
    }}"""
    source = tmp_path / "sigmoid.c"
    source.write_text(EMULATOR + kernel.source + reference)
    library = tmp_path / "sigmoid.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    dll.hybrid.argtypes = dll.reference.argtypes = [ctypes.c_void_p] * 2
    values = np.random.default_rng(742).uniform(-100, 100, elements).astype(np.float32)
    values[: min(elements, 7)] = np.array(
        [0, -0.0, 1, -1, np.inf, -np.inf, 1e-38], np.float32
    )[: min(elements, 7)]
    expected = np.empty(elements, np.float32)
    dll.reference(values.ctypes.data, expected.ctypes.data)
    for sentinel in (123.0, 456.0):
        output = np.full(elements + 32, sentinel, np.float32)
        dll.hybrid(values.ctypes.data, output[16:].ctypes.data)
        assert np.array_equal(output[16:-16].view(np.uint32), expected.view(np.uint32))
        assert np.all(output[:16] == sentinel) and np.all(output[-16:] == sentinel)
    assert all(
        (c.rs2.immediate & 15) == 5
        for c in commands
        if isinstance(c, Command) and c.funct == 33
    )


def test_hybrid_requires_an_explicit_f32_sigmoid_boundary():
    profile = load_profile(PROFILE)
    for dtype, operation, implementation in [
        ("bf16", "sigmoid", "f32_reciprocal"),
        ("f32", "rsqrt", "f32_reciprocal"),
        ("f32", "sigmoid", "unknown"),
    ]:
        with pytest.raises(ValueError):
            vpu_unary_kernel.emit_vpu_unary(
                profile,
                shape=(16,),
                dtype=dtype,
                operation=operation,
                symbol="kernel",
                sigmoid_implementation=implementation,
            )
