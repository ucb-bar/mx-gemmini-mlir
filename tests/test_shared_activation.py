"""Explicit prepared activations preserve actual quantizer bytes and readout."""

import ctypes
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.dense_contraction import plan_dense_shape
from mx_gemmini_support.dense_kernel import (
    emit_activation_preparation,
    emit_dense_f32_kernel,
    emit_shared_workspace_definition,
)
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


@pytest.mark.parametrize("precision", ["FP4", "FP8"])
@pytest.mark.parametrize("logical", [(1, 17, 37), (19, 23, 37), (32, 32, 64)])
def test_prepared_bytes_readout_and_changed_inputs(tmp_path, tmp_path_factory, precision, logical):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    profile = load_profile(PROFILE)
    plan = plan_dense_shape(profile, logical, precision=precision)
    issuer = emit_planned_runtime_issuer(profile, plan.padded_shape_mnk, plan.tile_mnk,
                                         precision=precision)
    kwargs = {"shared_workspace_symbol": "workspace",
              "prepacked_weight_symbols": ("fixed_codes", "fixed_scales")}
    regular = emit_dense_f32_kernel(issuer, logical, symbol="regular", **kwargs)
    prepared = emit_dense_f32_kernel(issuer, logical, symbol="prepared",
                                     prepared_activation=True, **kwargs)
    packing, size, scratch = emit_activation_preparation(issuer, logical, symbol="prepare")
    assert prepared.workspace_bytes < regular.workspace_bytes
    assert scratch == (0 if logical[0] == issuer.shape_mnk[0] and logical[2] == issuer.shape_mnk[2]
                       else issuer.shape_mnk[0] * issuer.shape_mnk[2] * 4)
    assert "mx_pack_activation_f32" not in prepared.source
    codes, scales = issuer.buffer_bytes["activation"], issuer.buffer_bytes["activation_scales"]
    m, n, _ = issuer.shape_mnk
    stub = f'''#include <stdint.h>
#include <string.h>
const uint8_t fixed_codes[{issuer.buffer_bytes['weight']}]={{0}};
const uint8_t fixed_scales[{issuer.buffer_bytes['weight_scales']}]={{0}};
uint8_t observed[{size}];
void mx_issue_encoded(const void *a,const void *as,const void *b,const void *bs,
                      const void *out,const void *scratch) {{
  (void)b; (void)bs; (void)scratch;
  memcpy(observed,a,{codes}); memcpy(observed+{codes},as,{scales});
  unsigned hash=0;
  for(unsigned i=0;i<{size};++i) hash=hash*33u+observed[i];
  for(unsigned i=0;i<{m*n};++i) ((uint16_t *)out)[i]=(uint16_t)(0x3f00+(hash%128)+(i%32));
}}
'''
    sources = {"regular": regular.source, "prepared": prepared.source, "packing": packing,
               "stub": stub, "workspace": emit_shared_workspace_definition(
                   regular.workspace_bytes, symbol="workspace")}
    paths = []
    for name, source in sources.items():
        path = tmp_path / (name + ".c")
        path.write_text(source)
        paths.append(str(path))
    # Keep dlopen inputs alive for the session: a suite that deletes each passed
    # tmp_path may reuse its inode while the native loader still caches the DSO.
    library = tmp_path_factory.mktemp("activation-dso") / "activation.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "mx_gemmini_support"), *paths,
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"), "-lm", "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    loaded.regular.argtypes = [pointer, pointer, pointer]
    loaded.prepare.argtypes = [pointer, pointer]
    loaded.prepared.argtypes = [pointer, pointer]
    rng = np.random.default_rng(937)
    packed = np.full(size, 0xA5, dtype=np.uint8)
    dummy = np.zeros((logical[2], logical[1]), dtype=np.float32)
    for extremes in (False, True):
        lhs = rng.normal(size=(logical[0], logical[2])).astype(np.float32)
        if extremes:
            lhs.flat[:5] = (0.0, -0.0, 1e-38, 1e30, -1e30)
        original = lhs.view(np.uint32).copy()
        expected = np.empty(logical[:2], dtype=np.float32)
        loaded.regular(lhs.ctypes.data, dummy.ctypes.data, expected.ctypes.data)
        observed = bytes((ctypes.c_uint8 * size).in_dll(loaded, "observed"))
        loaded.prepare(lhs.ctypes.data, packed.ctypes.data)
        assert packed.tobytes() == observed
        frozen = packed.copy()
        for _ in range(3):
            actual = np.empty_like(expected)
            loaded.prepared(packed.ctypes.data, actual.ctypes.data)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(packed, frozen)
        np.testing.assert_array_equal(lhs.view(np.uint32), original)
    # The existing matrix numeric contract traps on nonfinite operands. Check
    # unchanged refusal in children, preserving this test process and evidence.
    if logical == (1, 17, 37):
        for entry in ("regular", "prepare"):
            for value in ("nan", "inf", "-inf"):
                child = subprocess.run([sys.executable, "-c", '''
import ctypes,sys
import numpy as np
loaded=ctypes.CDLL(sys.argv[1])
lhs=np.full((1,37),float(sys.argv[3]),dtype=np.float32)
out=np.zeros(4096,dtype=np.uint8)
rhs=np.zeros((37,17),dtype=np.float32)
entry=getattr(loaded,sys.argv[2])
entry.argtypes=[ctypes.c_void_p]*(3 if sys.argv[2]=='regular' else 2)
if sys.argv[2]=='regular':entry(lhs.ctypes.data,rhs.ctypes.data,out.ctypes.data)
else:entry(lhs.ctypes.data,out.ctypes.data)
''', str(library), entry, value], capture_output=True, check=False)
                assert child.returncode == -signal.SIGILL
