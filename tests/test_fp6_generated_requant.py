"""The missing FP6 source drivers can bind generated, source-audited payloads."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from tools.generate_radiance_fp6_header import generate


SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))


@pytest.mark.parametrize("k", [128, 512])
def test_fp6_requant_header_generation_and_payload_binding(tmp_path, k):
    golden = SOURCE / "lib/golden/mx_golden"
    header = SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp6.m128n128k2048.h"
    if not golden.is_file() or not header.is_file():
        pytest.skip("requires checked-in Radiance FP6 header and built mx_golden")
    source_driver = SOURCE / "kernels/gemm_mxgemmini" / (
        f"mxgemm.fp6.singletile.tm128tn128tk{k}.requant.cpp")
    shutil.copy2(source_driver, tmp_path / source_driver.name)
    output = tmp_path / f"mxgemm.data.fp6.m128n128k{k}.h"
    receipt = generate(SOURCE, k, output)
    assert receipt["schema"] == "mx_gemmini.generated_radiance_fp6_fixture.v1"
    assert receipt["k"] == k
    assert len(receipt["generated_header_sha256"]) == 64
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate(SOURCE, k, output)
    kernel = read_source_gemm(tmp_path / source_driver.name)
    assert kernel.quant_output and kernel.shape == (128, 128, k)
    manifest = write_bundle(tmp_path / "bundle", kernel,
                            site_id="functional:matmul", profile_sha256="0" * 64)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    assert manifest["output_format"] == "fp6_e3m2"
    assert manifest["source_quant_golden_convention"] == "source_header"
    assert manifest["source_quant_header_format"] == "packed_fp6_lut_index"
    assert resources["golden_bf16"] and len(resources["golden_bf16"]) == 32768
    assert len(resources["source_fp6_packed"]) == 8192
    assert len(resources["golden_output_scales"]) == 512
    assert resources["source_fp6_packed"] != resources["nicolas_fp6"]
    assert resources["golden_output_scales"] != resources["nicolas_output_scales"]

    fullout = SOURCE / "kernels/gemm_mxgemmini" / (
        f"mxgemm.fp6.singletile.tm128tn128tk{k}.fullout.cpp")
    shutil.copy2(fullout, tmp_path / fullout.name)
    fullout_kernel = read_source_gemm(tmp_path / fullout.name)
    assert not fullout_kernel.quant_output and fullout_kernel.shape == (128, 128, k)
    fullout_manifest = write_bundle(tmp_path / "fullout_bundle", fullout_kernel,
                                    site_id="functional:matmul", profile_sha256="0" * 64)
    checked_fullout, fullout_resources = load_bundle(tmp_path / "fullout_bundle")
    assert checked_fullout == fullout_manifest
    assert fullout_manifest.get("output_format") is None
    assert fullout_resources["golden_bf16"] == resources["golden_bf16"]


@pytest.mark.parametrize("k", [256, 1024])
def test_fp6_multitile_fullout_generated_header_and_payload(tmp_path, k):
    golden = SOURCE / "lib/golden/mx_golden"
    header = SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp6.m128n128k2048.h"
    if not golden.is_file() or not header.is_file():
        pytest.skip("requires checked-in Radiance FP6 header and built mx_golden")
    driver = SOURCE / "kernels/gemm_mxgemmini" / (
        f"mxgemm.fp6.m128n128k{k}.tm128tn128tk{128 if k == 256 else 512}.fullout.cpp")
    shutil.copy2(driver, tmp_path / driver.name)
    generated = tmp_path / f"mxgemm.data.fp6.m128n128k{k}.h"
    receipt = generate(SOURCE, k, generated)
    assert receipt["generated_header_sha256"] and receipt["k"] == k
    kernel = read_source_gemm(tmp_path / driver.name)
    assert not kernel.quant_output and kernel.shape == (128, 128, k)
    manifest = write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256="0" * 64)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    assert len(resources["golden_bf16"]) == 32768
    assert len(resources["activation"]) == 128 * k // 2
    assert len(resources["activation_lut"]) == 64 * 12
