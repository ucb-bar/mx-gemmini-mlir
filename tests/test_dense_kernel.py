"""The dense catalog entry packs live tensors and preserves row-major output."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from mx_gemmini_support.dense_kernel import (emit_dense_f32_batch_kernel,
                                             emit_dense_f32_kernel,
                                             emit_shared_workspace_definition)
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.runtime_issuer import RuntimeIssuer, emit_runtime_issuer
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
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


def test_batched_dense_wrapper_advances_disjoint_slices(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    profile = load_profile(PROFILE)
    issuer = emit_planned_runtime_issuer(
        profile, (32, 32, 32), (32, 32, 32), precision="FP4")
    batch_kernel = emit_dense_f32_batch_kernel(
        issuer, (3, 8, 28, 32), symbol="mx_batch_probe")
    (tmp_path / "adapter.c").write_text(batch_kernel.source)
    (tmp_path / "stub.c").write_text('''#include <stdint.h>
unsigned calls = 0;
void mx_issue_encoded(const void *a, const void *as, const void *b, const void *bs,
                      const void *out, const void *scratch) {
  (void)a; (void)as; (void)b; (void)bs; (void)scratch;
  ++calls;
  uint16_t *words = (uint16_t *)out;
  for (unsigned i = 0; i < 32 * 32; ++i) words[i] = (uint16_t)(0x3f80 + calls);
}
''')
    library = tmp_path / "batch.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
                    "-Werror", "-I", str(ROOT / "mx_gemmini_support"),
                    str(tmp_path / "adapter.c"), str(tmp_path / "stub.c"),
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"),
                    "-lm", "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    loaded.mx_batch_probe.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    a = np.ones((3, 8, 32), dtype=np.float32)
    b = np.ones((3, 32, 28), dtype=np.float32)
    out = np.empty((3, 8, 28), dtype=np.float32)
    loaded.mx_batch_probe(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert ctypes.c_uint.in_dll(loaded, "calls").value == 3
    for index in range(3):
        assert np.all(out[index].view(np.uint32) == (0x3f80 + index + 1) << 16)


def test_two_catalog_entries_share_one_aligned_serial_workspace(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    issuer = emit_planned_runtime_issuer(
        load_profile(PROFILE), (32, 32, 32), (32, 32, 32), precision="FP4")
    kernels = [emit_dense_f32_kernel(
        issuer, (8, 28, 32), symbol=f"mx_entry_{index}",
        shared_workspace_symbol="mx_catalog_workspace") for index in range(2)]
    assert kernels[0].workspace_bytes == kernels[1].workspace_bytes
    assert "static uint8_t activation" not in kernels[0].source
    for index, kernel in enumerate(kernels):
        (tmp_path / f"entry_{index}.c").write_text(kernel.source)
    (tmp_path / "workspace.c").write_text(emit_shared_workspace_definition(
        max(kernel.workspace_bytes for kernel in kernels), symbol="mx_catalog_workspace"))
    (tmp_path / "stub.c").write_text('''#include <stdint.h>
unsigned calls = 0;
void mx_issue_encoded(const void *a, const void *as, const void *b, const void *bs,
                      const void *out, const void *scratch) {
  (void)a; (void)as; (void)b; (void)bs; (void)scratch;
  ++calls;
  uint16_t *words = (uint16_t *)out;
  for (unsigned i = 0; i < 32 * 32; ++i) words[i] = (uint16_t)(0x3f80 + calls);
}
''')
    library = tmp_path / "catalog.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
                    "-Werror", "-I", str(ROOT / "mx_gemmini_support"),
                    *(str(tmp_path / f"entry_{index}.c") for index in range(2)),
                    str(tmp_path / "workspace.c"), str(tmp_path / "stub.c"),
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"),
                    "-lm", "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    lhs = np.ones((8, 32), dtype=np.float32)
    rhs = np.ones((32, 28), dtype=np.float32)
    for index in range(2):
        output = np.empty((8, 28), dtype=np.float32)
        entry = getattr(loaded, f"mx_entry_{index}")
        entry.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
        entry(lhs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        assert np.all(output.view(np.uint32) == (0x3f80 + index + 1) << 16)
    assert ctypes.c_uint.in_dll(loaded, "calls").value == 2


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
