"""Source quantized-output resources remain explicit and byte-bound."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, read_source_payload, write_bundle
from mx_gemmini_support.quant_reference import (
    quantize_bf16_fp4_output, quantize_bf16_fp8_output)


def test_nicolas_fp8_reference_uses_po2_scale_and_signed_rne_codes():
    words = [0x3f80, 0xbf80] + [0] * 62
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    codes, scales = quantize_bf16_fp8_output(raw, 1, 64)
    assert list(codes[:2]) == [0x38, 0xb8]
    assert codes[2:] == bytes(62)
    assert list(scales) == [127, 104]


def test_nicolas_fp4_reference_packs_even_and_odd_rows():
    words = [0x3f80] + [0] * 31 + [0xbf80] + [0] * 31
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    codes, scales = quantize_bf16_fp4_output(raw, 2, 32)
    assert codes[0] == 0xa2
    assert codes[1:] == bytes(31)
    assert list(scales) == [127, 127]
    empty_codes, empty_scales = quantize_bf16_fp4_output(bytes(2 * 2 * 32), 2, 32)
    assert empty_codes == bytes(32)
    assert list(empty_scales) == [0, 0]


SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
MXQ = Path(os.environ.get("MXQ_ROOT", "/nonexistent"))


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


def test_fp4_requant_output_records_source_fp8_format_and_packed_target(tmp_path):
    driver = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp4.singletile.tm64tn64tk64.requant.cpp"
    if not driver.is_file() or not (SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp4.m64n64k64.h").is_file():
        pytest.skip("requires generated Radiance FP4 64x64x64 header")
    kernel = read_source_gemm(driver)
    assert kernel.quant_output
    manifest = write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256="0" * 64)
    _, resources = load_bundle(tmp_path / "bundle")
    assert manifest["output_format"] == "fp4_e2m1"
    assert manifest["source_quant_header_format"] == "fp8_e4m3"
    assert len(resources["golden_fp8"]) == 64 * 64
    assert len(resources["nicolas_fp4"]) == 64 * 64 // 2
    assert len(resources["nicolas_output_scales"]) == 64 * 2


def test_fp4_oracle_matches_pinned_mxquant_for_full_source_output():
    driver = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp4.singletile.tm64tn64tk64.requant.cpp"
    header = SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp4.m64n64k64.h"
    if not driver.is_file() or not header.is_file() or not MXQ.is_dir():
        pytest.skip("requires generated FP4 source header and pinned MXQuant")
    revision = subprocess.check_output(["git", "-C", str(MXQ), "rev-parse", "HEAD"],
                                       text=True).strip()
    assert revision == "b4af5430bac147f4a16126931cc0177367cc3982"
    sys.path.insert(0, str(MXQ))
    import torch
    from mxq.block.mxgemmini import quantize
    from mxq.scale_factor import HARDWARE_FLOOR

    source = read_source_payload(read_source_gemm(driver))
    values = torch.frombuffer(bytearray(source["golden_bf16"].data),
                              dtype=torch.uint16).view(torch.bfloat16).reshape(64, 64).float()
    codes, scales = quantize(values, "MXFP4_E2M1", axis=1, rounding_mode="rne",
                             scale_floor=HARDWARE_FLOOR, via=(3, 1))
    grid = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}
    packed = bytearray(64 * 64 // 2)
    for row, line in enumerate(codes.tolist()):
        for col, value in enumerate(line):
            code = grid[abs(value)] | (8 if value < 0 else 0)
            packed[row // 2 * 64 + col] |= code << (4 if row & 1 else 0)
    expected_scales = bytes((torch.log2(scales).to(torch.int16) + 127).flatten().tolist())
    assert bytes(packed) == source["nicolas_fp4"].data
    assert expected_scales == source["nicolas_output_scales"].data
