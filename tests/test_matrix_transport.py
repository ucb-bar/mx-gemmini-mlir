"""Matrix readout and mixed VPU boundaries retain bits and refusal policy."""

import ctypes
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.dense_contraction import plan_dense_shape
from mx_gemmini_support.dense_kernel import emit_dense_f32_kernel
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def library(tmp_path, sources):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler required")
    paths = []
    for name, source in sources.items():
        path = tmp_path / (name + ".c")
        path.write_text(source)
        paths.append(str(path))
    output = tmp_path / "transport.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(ROOT / "mx_gemmini_support"),
                    *paths, str(ROOT / "mx_gemmini_support/runtime_quant.c"), "-o", str(output)], check=True)
    return output, ctypes.CDLL(str(output))


@pytest.mark.parametrize("precision", ["FP4", "FP8"])
@pytest.mark.parametrize("logical", [(32, 32, 64), (16, 32, 64), (1, 17, 37)])
def test_bf16_readout_matches_widened_words_and_crops_safely(tmp_path, precision, logical):
    plan = plan_dense_shape(load_profile(PROFILE), logical, precision=precision)
    issuer = emit_planned_runtime_issuer(load_profile(PROFILE), plan.padded_shape_mnk, plan.tile_mnk,
                                         precision=precision)
    kwargs = {"prepacked_weight_symbols": ("fixed_codes", "fixed_scales")}
    original = emit_dense_f32_kernel(issuer, logical, symbol="original", **kwargs)
    compact = emit_dense_f32_kernel(issuer, logical, symbol="compact", output_dtype="bf16", **kwargs)
    m, n, _ = issuer.shape_mnk
    stub = f'''#include <stdint.h>
const uint8_t fixed_codes[{issuer.buffer_bytes['weight']}]={{0}};
const uint8_t fixed_scales[{issuer.buffer_bytes['weight_scales']}]={{0}};
uintptr_t last_output;
void mx_issue_encoded(const void *a,const void *as,const void *b,const void *bs,
                      const void *out,const void *scratch) {{
  (void)a;(void)as;(void)b;(void)bs;(void)scratch;
  last_output=(uintptr_t)out;
  for(unsigned i=0;i<{m*n};++i) ((uint16_t *)out)[i]=(uint16_t)(i*97u);
}}
'''
    _, loaded = library(tmp_path, {"original": original.source, "compact": compact.source, "stub": stub})
    for name in ("original", "compact"):
        getattr(loaded, name).argtypes = [ctypes.c_void_p] * 3
    lhs = np.ones((logical[0], logical[2]), dtype=np.float32)
    rhs = np.ones((logical[2], logical[1]), dtype=np.float32)
    old = np.empty(logical[:2], dtype=np.float32)
    backing = np.full(logical[0] * logical[1] + 16, 0xA57E, dtype=np.uint16)
    new = backing[:old.size].reshape(old.shape)
    loaded.original(lhs.ctypes.data, rhs.ctypes.data, old.ctypes.data)
    loaded.compact(lhs.ctypes.data, rhs.ctypes.data, new.ctypes.data)
    np.testing.assert_array_equal(old.view(np.uint32), new.astype(np.uint32) << 16)
    assert np.all(backing[old.size:] == 0xA57E)
    direct = logical[:2] == issuer.shape_mnk[:2]
    assert (ctypes.c_size_t.in_dll(loaded, "last_output").value == new.ctypes.data) == direct
    if direct:
        assert original.workspace_bytes - compact.workspace_bytes == issuer.buffer_bytes["output_bf16"]


def test_compact_vpu_boundary_preserves_all_non_nan_codes_and_nan_refusal(tmp_path, monkeypatch):
    from mx_gemmini_support import vpu_add_kernel

    def stub(profile, *, elements, symbol, operation):
        # Boundary-only observer; actual device arithmetic is qualified by Spike.
        return f'''static void {symbol}(const void *a,const void *b,const void *c) {{
          (void)b;for(unsigned i=0;i<{elements};++i) ((uint16_t *)c)[i]=((const uint16_t *)a)[i];
        }}'''

    monkeypatch.setattr(vpu_add_kernel, "_issue", stub)
    all_codes = np.arange(65536, dtype=np.uint16)
    valid = all_codes[((all_codes & 0x7F80) != 0x7F80) | ((all_codes & 0x7F) == 0)]
    shape = (valid.size,)
    profile = load_profile(PROFILE)
    old = vpu_add_kernel.emit_vpu_elementwise(profile, result_shape=shape, rhs_shape=shape,
                                              dtype="f32", operation="add", symbol="old_boundary")
    new = vpu_add_kernel.emit_vpu_elementwise(profile, result_shape=shape, rhs_shape=shape,
                                              dtype="f32", operation="add", symbol="new_boundary",
                                              lhs_dtype="bf16", preserve_f32_nan_refusal=True)
    path, loaded = library(tmp_path, {"old": old.source, "new": new.source})
    for name in ("old_boundary", "new_boundary"):
        getattr(loaded, name).argtypes = [ctypes.c_void_p] * 3
    widened = (valid.astype(np.uint32) << 16).view(np.float32)
    rhs = np.zeros_like(widened)
    a, b = np.empty_like(widened), np.empty_like(widened)
    loaded.old_boundary(widened.ctypes.data, rhs.ctypes.data, a.ctypes.data)
    loaded.new_boundary(valid.ctypes.data, rhs.ctypes.data, b.ctypes.data)
    np.testing.assert_array_equal(a.view(np.uint32), widened.view(np.uint32))
    np.testing.assert_array_equal(b.view(np.uint32), widened.view(np.uint32))
    for name in ("old_boundary", "new_boundary"):
        for word in (0x7F81, 0x7FC0, 0xFF81):
            child = subprocess.run([sys.executable, "-c", '''
import ctypes,sys,numpy as np
lib=ctypes.CDLL(sys.argv[1]); name=sys.argv[2]; code=int(sys.argv[3]); size=int(sys.argv[4])
raw=np.full(size,code,dtype=np.uint16)
lhs=raw if name=='new_boundary' else (raw.astype(np.uint32)<<16).view(np.float32)
rhs=np.zeros(size,dtype=np.float32); out=np.zeros_like(rhs)
f=getattr(lib,name); f.argtypes=[ctypes.c_void_p]*3
f(lhs.ctypes.data,rhs.ctypes.data,out.ctypes.data)
''', str(path), name, str(word), str(valid.size)], capture_output=True, check=False)
            assert child.returncode == -signal.SIGILL
