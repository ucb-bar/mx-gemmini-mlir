"""Source quantized-output resources remain explicit and byte-bound."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.quant_reference import quantize_bf16_fp8_output


def test_nicolas_fp8_reference_uses_po2_scale_and_signed_rne_codes():
    words = [0x3f80, 0xbf80] + [0] * 62
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    codes, scales = quantize_bf16_fp8_output(raw, 1, 64)
    assert list(codes[:2]) == [0x38, 0xb8]
    assert codes[2:] == bytes(62)
    assert list(scales) == [127, 104]


SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))


def test_fp8_requant_output_codes_and_scales_are_bound(tmp_path):
    driver = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp8.singletile.tm64tn64tk64.requant.cpp"
    if not driver.is_file() or not (SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp8.m64n64k64.h").is_file():
        pytest.skip("requires generated Radiance FP8 64x64x64 header")
    kernel = read_source_gemm(driver)
    assert kernel.quant_output
    manifest = write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256="0" * 64)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    assert manifest["output_format"] == "fp8_e4m3"
    assert len(resources["golden_fp8"]) == 64 * 64
    assert len(resources["golden_output_scales"]) == 64 * 2
    assert len(resources["nicolas_fp8"]) == 64 * 64
    assert len(resources["nicolas_output_scales"]) == 64 * 2
    assert resources["nicolas_output_scales"] != resources["golden_output_scales"]
    raw = resources["output_scales"]
    for row in range(64):
        for group in range(2):
            assert resources["golden_output_scales"][row * 2 + group] == raw[group * 64 + row]


def test_unsupported_requant_output_format_is_rejected(tmp_path):
    driver = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp4.singletile.tm64tn64tk64.requant.cpp"
    if not driver.is_file() or not (SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp4.m64n64k64.h").is_file():
        pytest.skip("requires generated Radiance FP4 64x64x64 header")
    kernel = read_source_gemm(driver)
    assert kernel.quant_output
    with pytest.raises(ValueError, match="qualified only for FP8"):
        write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                     profile_sha256="0" * 64)
