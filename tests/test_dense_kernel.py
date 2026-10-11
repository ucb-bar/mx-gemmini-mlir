"""The dense catalog entry packs live tensors and preserves row-major output."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from mx_gemmini_support.dense_kernel import emit_dense_f32_kernel
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.runtime_issuer import RuntimeIssuer, emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _issuer(site: str) -> RuntimeIssuer:
    archive = ARCHIVE / site
    manifest, resources = load_bundle(archive / "bundle")
    profile = load_profile(PROFILE)
    program = lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                                 profile, manifest, resources)
    return emit_runtime_issuer(program, manifest)


@pytest.mark.parametrize("site", ["g", "dn"])
def test_dense_wrapper_packs_archived_model_operands_and_unpads_output(tmp_path, site: str) -> None:
    np = pytest.importorskip("numpy")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    a = np.load(ARCHIVE / "inputs" / f"{site}.activation.npy", allow_pickle=False)
    b = np.load(ARCHIVE / "inputs" / f"{site}.weight.npy", allow_pickle=False)
    logical = (a.shape[0], b.shape[1], a.shape[1])
    issuer = _issuer(site)
    kernel = emit_dense_f32_kernel(issuer, logical, symbol="mx_dense_probe")
    assert kernel.workspace_bytes > sum(issuer.buffer_bytes.values())
    (tmp_path / "adapter.c").write_text(kernel.source)
    m, n, k = issuer.shape_mnk
    stub = f"""#include <stdint.h>
const uint8_t *observed_a, *observed_as, *observed_b, *observed_bs;
void mx_issue_encoded(const void *a, const void *as, const void *b, const void *bs,
                      const void *out, const void *scratch) {{
  (void)scratch;
  observed_a = a; observed_as = as; observed_b = b; observed_bs = bs;
  uint16_t *words = (uint16_t *)out;
  for (unsigned row = 0; row < {m}; ++row)
    for (unsigned col = 0; col < {n}; ++col)
      words[row * {n} + col] = (uint16_t)(0x3f80 + row * {n} + col);
}}
"""
    (tmp_path / "stub.c").write_text(stub)
    library = tmp_path / "adapter.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
                    "-Werror", "-I", str(ROOT / "mx_gemmini_support"),
                    str(tmp_path / "adapter.c"), str(tmp_path / "stub.c"),
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"),
                    "-lm", "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    loaded.mx_dense_probe.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    result = np.empty((logical[0], logical[1]), dtype=np.float32)
    loaded.mx_dense_probe(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    bundle = ARCHIVE / site / "bundle"
    for name, exported in (("activation", "observed_a"),
                           ("activation_scales", "observed_as"),
                           ("weight", "observed_b"),
                           ("weight_scales", "observed_bs")):
        pointer = ctypes.c_void_p.in_dll(loaded, exported).value
        assert pointer is not None
        assert ctypes.string_at(pointer, issuer.buffer_bytes[name]) == (
            bundle / f"{name}.bin").read_bytes()
    expected = np.array([[((0x3f80 + row * n + col) << 16)
                          for col in range(logical[1])]
                         for row in range(logical[0])], dtype=np.uint32)
    np.testing.assert_array_equal(result.view(np.uint32), expected)


def test_dense_wrapper_refuses_changed_extent_or_shape() -> None:
    issuer = _issuer("g")
    with pytest.raises(ValueError, match="exceed"):
        emit_dense_f32_kernel(issuer, (issuer.shape_mnk[0] + 1, 1, 1), symbol="entry")
    from dataclasses import replace

    changed = replace(issuer, buffer_bytes={**issuer.buffer_bytes, "activation": 1})
    with pytest.raises(ValueError, match="extents"):
        emit_dense_f32_kernel(changed, (8, 344, 128), symbol="entry")


def test_direct_fp8_schedule_uses_fp8_packing_without_weight_padding() -> None:
    archive = ROOT / "docs/evidence/model2mlir_tinyllama_projection_a042643_266c593"
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    manifest, resources = load_bundle(archive / "bundle")
    program = lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                                 profile, manifest, resources)
    issuer = emit_runtime_issuer(program, manifest)
    kernel = emit_dense_f32_kernel(issuer, (8, 32, 2048), symbol="mx_fp8_probe")
    assert "MX_RUNTIME_FP8" in kernel.source
    assert "static float padded_a" in kernel.source
    assert "static float padded_b" not in kernel.source
    assert kernel.workspace_bytes == sum(issuer.buffer_bytes.values()) + 16 * 2048 * 4
