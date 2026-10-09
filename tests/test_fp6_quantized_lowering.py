"""FP6 LUT output qualification from the checked-in Radiance fullout data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.layout import pack_fp6_lut
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import quantize_bf16_fp6_lut_output
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
DRIVER = SOURCE / ("kernels/gemm_mxgemmini/"
                   "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp")


def test_fp6_lut_reference_packs_row_pairs_and_zero_scales():
    line = (0, 12, 44) + (0,) * 13  # zero, +1, -1 in E3M2
    lut = pack_fp6_lut([line] * 64)
    words = [0x3f80] + [0] * 31 + [0xbf80] + [0] * 31
    bf16 = b"".join(word.to_bytes(2, "little") for word in words)
    codes, scales = quantize_bf16_fp6_lut_output(bf16, 2, 32, lut)
    assert codes[0] == 0x21
    assert codes[1:] == bytes(31)
    assert scales == bytes([127, 127])
    empty, empty_scales = quantize_bf16_fp6_lut_output(bytes(2 * 2 * 32), 2, 32, lut)
    assert empty == bytes(32)
    assert empty_scales == bytes(2)


def test_fp6_fullout_source_specialization_delays_quantization_to_final_wave(tmp_path):
    if not DRIVER.is_file() or not DRIVER.with_name(
            "mxgemm.data.fp6.m128n128k2048.h").is_file() or not RTL.is_dir():
        pytest.skip("requires checked-in Radiance FP6 header and Nicolas RTL")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(
        tmp_path / "bundle", read_source_gemm(DRIVER),
        site_id="functional:matmul", profile_sha256=profile_sha256(profile),
        fp6_quantized_specialization=True)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    assert manifest["output_specialization"] == "bf16_fullout_to_fp6_lut_quantized"
    assert len(resources["source_fp6_packed"]) == 8192
    assert len(resources["nicolas_fp6"]) == 8192
    assert len(resources["nicolas_output_scales"]) == 512
    bound = bind_payload(
        (ROOT / "docs/evidence/model2mlir_radiance_mx_fp6_bound_20261009.mlir").read_text(),
        profile, manifest)
    assert 'mx.output_specialization = "source_bf16_fp6_lut_quantized"' in bound
    assert '"mx_gemmini.readout_quantized"' in bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.output_format == "fp6_e3m2"
    assert len(program.plan["waves"]) == 16
    configs = [(step.phase, step.command.rs1.immediate)
               for step in program.steps if isinstance(step.command, Command) and
               step.command.funct == 0 and step.command.rs1.immediate is not None and
               step.command.rs1.immediate & 0xffff0000 == 0x10000]
    assert (configs[0][1] >> 14) & 3 == 3  # intermediate BF16
    assert configs[-1][0] == "configure_final_output"
    assert (configs[-1][1] >> 14) & 3 == 1  # final FP6 E3M2
    receipt = write_standalone_sources(tmp_path / "artifact", program, resources)
    assert receipt["source_quant_code_differences"] > 0
    assert receipt["source_quant_scale_differences"] == 512
    assert "FP6 packed-index mismatches" in (tmp_path / "artifact/mx_driver.c").read_text()


def test_fp6_quantized_bundle_rejects_modified_lut(tmp_path):
    if not DRIVER.is_file() or not DRIVER.with_name(
            "mxgemm.data.fp6.m128n128k2048.h").is_file():
        pytest.skip("requires checked-in Radiance FP6 header")
    write_bundle(tmp_path / "bundle", read_source_gemm(DRIVER),
                 site_id="functional:matmul", profile_sha256="0" * 64,
                 fp6_quantized_specialization=True)
    path = tmp_path / "bundle/output_lut.bin"
    changed = bytearray(path.read_bytes())
    changed[0] ^= 1
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="digest or size"):
        load_bundle(tmp_path / "bundle")
