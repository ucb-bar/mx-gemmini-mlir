"""FP32 normalization retains its declared reduction and canonical packed bytes."""

import ctypes
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.f32_rms_packing import emit_f32_rms_packing
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)


@pytest.mark.parametrize(
    "width,reduction",
    [
        (32, "ordered"),
        (256, "ordered"),
        (2048, "ordered"),
        (256, "striped16"),
        (2048, "striped16"),
    ],
)
def test_packed_bytes_equal_independent_reduction_and_scalar_quantizer(
    tmp_path, width, reduction
):
    rows = 16
    kernel = emit_f32_rms_packing(
        load_profile(PROFILE),
        rows=rows,
        width=width,
        symbol="packed",
        workspace_symbol="workspace",
        reduction=reduction,
    )
    archive_dir = ROOT / "docs/evidence/mx_compiler_optimization_probes_20261019"
    manifest = json.loads((archive_dir / "artifacts.json").read_text())
    with tarfile.open(archive_dir / "probes.tar.xz", "r:xz") as archive:
        reference_quant = archive.extractfile("fp8_pack/old_quant.c").read()
    assert (
        hashlib.sha256(reference_quant).hexdigest()
        == manifest["members"]["fp8_pack/old_quant.c"]["sha256"]
    )
    (tmp_path / "reference_quant.c").write_bytes(reference_quant)
    (tmp_path / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    if reduction == "ordered":
        summation = (
            f"for(unsigned col=0;col<{width};++col)sum=sum+square[row*{width}+col];"
        )
    else:
        summation = f"""float sums[16]={{0}};
        for(unsigned lane=0;lane<16;++lane)
          for(unsigned word=lane;word<{width // 2};word+=128)
            for(unsigned j=0;j<8;++j){{
              unsigned col=2*(word+16*j);
              sums[lane]+=square[row*{width}+col]+square[row*{width}+col+1];
            }}
        for(unsigned lane=0;lane<16;++lane)sum+=sums[lane];"""
    reference = f"""
unsigned char workspace[{kernel.workspace_bytes + 64}] __attribute__((aligned(64)));
void reference(const float *x,const float *gamma,const float *d,const float *eps,float *out){{
 float square[{rows * width}],inverse[{rows}],first[{rows * width}];
 for(unsigned i=0;i<{rows * width};++i)square[i]=x[i]*x[i];
 for(unsigned row=0;row<{rows};++row){{float sum=0.0f;{summation}
   inverse[row]=1.0f/__builtin_sqrtf(sum/(*d)+*eps);}}
 for(unsigned i=0;i<{rows * width};++i)first[i]=x[i]*inverse[i/{width}];
 for(unsigned i=0;i<{rows * width};++i)out[i]=gamma[i%{width}]*first[i];
}}
"""
    source = tmp_path / "rms.c"
    source.write_text(kernel.source + reference)
    library = tmp_path / "rms.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DMX_QUANT_REFERENCE",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "mx_gemmini_support"),
            str(source),
            str(tmp_path / "reference_quant.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    dll.packed.argtypes = dll.reference.argtypes = [ctypes.c_void_p] * 5
    dll.mx_pack_activation_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(119)
    x = rng.normal(size=(rows, width)).astype(np.float32)
    x[0] = 0
    x[1] = -np.float32(0)
    x[2] *= np.float32(1e-20)
    x[3] *= np.float32(1e18)
    gamma = rng.uniform(-1, 2, width).astype(np.float32)
    gamma[0] = 0
    divisor, eps = np.array(width, np.float32), np.array(1e-5, np.float32)
    workspace = np.ctypeslib.as_array(
        (ctypes.c_uint8 * (kernel.workspace_bytes + 64)).in_dll(dll, "workspace")
    )
    for factor in (1, 0.5):
        x *= np.float32(factor)
        gamma += np.float32(0.125)
        original = x.copy().view(np.uint32)
        reference = np.empty_like(x)
        dll.reference(
            x.ctypes.data,
            gamma.ctypes.data,
            divisor.ctypes.data,
            eps.ctypes.data,
            reference.ctypes.data,
        )
        expected = np.empty(kernel.packed_bytes, np.uint8)
        assert (
            dll.mx_pack_activation_f32(
                reference.ctypes.data,
                rows,
                width,
                8,
                expected.ctypes.data,
                expected[rows * width :].ctypes.data,
            )
            == 0
        )
        output = np.full(kernel.packed_bytes + 128, 0xA5, np.uint8)
        workspace[:] = 0xA5
        dll.packed(
            x.ctypes.data,
            gamma.ctypes.data,
            divisor.ctypes.data,
            eps.ctypes.data,
            output[64:].ctypes.data,
        )
        assert np.array_equal(output[64:-64], expected)
        assert (
            np.all(output[:64] == 0xA5)
            and np.all(output[-64:] == 0xA5)
            and np.all(workspace[-64:] == 0xA5)
        )
        assert np.array_equal(x.view(np.uint32), original)


@pytest.mark.parametrize(
    "rows,width,reduction",
    [
        (1, 2048, "ordered"),
        (16, 33, "ordered"),
        (16, 32, "striped16"),
        (16, 256, "unknown"),
        (16, 8224, "ordered"),
    ],
)
def test_unsupported_row_layout_or_reduction_is_refused(rows, width, reduction):
    with pytest.raises(ValueError):
        emit_f32_rms_packing(
            load_profile(PROFILE),
            rows=rows,
            width=width,
            symbol="packed",
            reduction=reduction,
        )
