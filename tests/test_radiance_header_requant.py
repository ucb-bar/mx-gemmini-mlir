"""Source-header FP8 semantics lower through BF16 MX readout and a host epilogue."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import quantize_bf16_radiance_header_fp8
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))


def test_radiance_header_reference_uses_legacy_scale_and_zero_floor():
    words = [0x3f80, 0xbf80, 0x3780] + [0] * 29
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    codes, scales = quantize_bf16_radiance_header_fp8(raw, 1, 32)
    assert scales == bytes([119])
    assert codes[:3] == bytes([0x78, 0xf8, 0])


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_source_header_epilogue_is_typed_and_byte_exact(tmp_path, precision):
    driver = SOURCE / ("kernels/gemm_mxgemmini/"
                       f"mxgemm.{precision}.singletile.tm64tn64tk64.requant.cpp")
    if not driver.is_file() or not RTL.is_dir():
        pytest.skip("requires Radiance source and Nicolas RTL")
    kernel = read_source_gemm(driver)
    if not kernel.data_header_present:
        pytest.skip("requires generated source quantized header")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(tmp_path / "bundle")
    codes, scales = quantize_bf16_radiance_header_fp8(resources["golden_bf16"], 64, 64)
    assert (codes, scales) == (resources["golden_fp8"],
                               resources["golden_output_scales"])

    captured = (ROOT / "docs/evidence" /
                f"model2mlir_radiance_mx_{precision}_64x64x64_quant_bound.mlir").read_text()
    bound = bind_payload(captured, profile, manifest, source_header_quantized=True)
    assert '"mx_gemmini.readout_bf16"' in bound
    assert '"mx_gemmini.host_requantize"' in bound
    assert '"mx_gemmini.readout_quantized"' not in bound
    assert verify_ir(bound, profile)["source_resources"] == 4
    mx_opt = ROOT / "build/tools/mx-gemmini-opt"
    if mx_opt.is_file():
        subprocess.run([str(mx_opt), "-o", "/dev/null"], input=bound, text=True, check=True)

    physical = lower_bound_source(bound, profile, manifest, resources)
    assert physical.output_format == "radiance_header_fp8"
    assert physical.source_golden_preserving
    assert all(step.phase != "configure_final_output" for step in physical.steps)
    receipt = write_standalone_sources(tmp_path / "standalone", physical, resources)
    assert receipt["golden_basis"] == "radiance_header_fp8_from_mx_bf16"
    driver_c = (tmp_path / "standalone/mx_driver.c").read_text()
    assert "radiance_header_requantize" in driver_c
    assert "log2f" not in driver_c and "ldexpf" not in driver_c

    with pytest.raises(ValueError, match="Radiance FP8 header policy"):
        verify_ir(bound.replace('quant_policy = "radiance_header_fp8_v1"',
                                'quant_policy = "unknown"'), profile)
    bad_shape = bound.replace(
            '-> (tensor<64x64xi8>, tensor<64x2xi8>)',
            '-> (tensor<32x64xi8>, tensor<64x2xi8>)').replace(
            'func.return %4, %5 : tensor<64x64xi8>, tensor<64x2xi8>',
            'func.return %4, %5 : tensor<32x64xi8>, tensor<64x2xi8>')
    with pytest.raises(ValueError, match="result shape differs"):
        verify_ir(bad_shape, profile)
    bad = dict(resources)
    bad["golden_fp8"] = bytes([0]) + resources["golden_fp8"][1:]
    if bad["golden_fp8"] == resources["golden_fp8"]:
        bad["golden_fp8"] = bytes([1]) + resources["golden_fp8"][1:]
    with pytest.raises(ValueError, match="golden differs"):
        lower_bound_source(bound, profile, manifest, bad)
