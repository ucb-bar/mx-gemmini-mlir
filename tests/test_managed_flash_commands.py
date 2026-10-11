"""Pinned RTL's gated load and managed scale-half protocol for flash attention."""

from pathlib import Path
import os
import subprocess

import pytest

from mx_gemmini_support.memory_phase import (lower_matrix_mvin,
                                             lower_strided_scale_dma)
from mx_gemmini_support.nicolas_attention_matrix import lower_managed_spad_matmul
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.target_profile import profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def test_mid_flash_first_qk_and_pv_scale_halves_match_source_macros() -> None:
    profile = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    # attn_flash_mid.c: Sq=64, Sk=128, D=64, Bk=32. KTS is
    # [D/32][Sk], VS is [Sk/32][D]. The first QK and PV use halves 0,1.
    qk_scale = lower_strided_scale_dma(
        profile, buffer="KT_SCALES", row_bytes=32, rows=2,
        pitch_bytes=128, destination_byte=0, selector=1, gated=True)
    pv_scale = lower_strided_scale_dma(
        profile, buffer="V_SCALES", row_bytes=64, rows=1,
        pitch_bytes=64, destination_byte=4096, selector=1, gated=True)
    assert qk_scale.minimum_buffer_bytes == 160
    assert qk_scale.commands[0].funct == pv_scale.commands[0].funct == 27
    assert qk_scale.commands[0].rs1.or_bits == 128 << 40
    assert qk_scale.commands[0].rs2.immediate == (
        (1 << 54) | (2 << 46) | (1 << 32) | 32)
    assert pv_scale.commands[0].rs2.immediate == (
        (1 << 54) | (1 << 46) | (4096 << 33) | (1 << 32) | 64)

    qk = lower_managed_spad_matmul(
        profile, m=64, n=32, k=64, a_row=0, b_end_row=4224,
        output_row=8192, weight_half=0, activation_half=1,
        scale_sink="scale_sink")
    pv = lower_managed_spad_matmul(
        profile, m=64, n=64, k=32, a_row=4608, b_end_row=384,
        output_row=13056, weight_half=1, activation_half=0,
        scale_sink="scale_sink")
    assert [command.funct for command in qk] == [26, 9, 24, 8]
    assert [command.funct for command in pv] == [26, 9, 24, 8]
    assert qk[0].rs1.or_bits == ((1 << 60) | (4 << 51) | (2 << 42) | (4 << 33))
    assert pv[0].rs1.or_bits == ((1 << 61) | (2 << 51) | (4 << 42) | (4 << 33))
    assert qk[0].rs2.immediate == pv[0].rs2.immediate == 0x60001
    assert qk[3].rs2.immediate == (8192 << 32) | 0x338
    assert pv[3].rs2.immediate == (13056 << 32) | 0x338

    kt_block = lower_matrix_mvin(
        profile, buffer="KT_IN", matrix_rows=64, matrix_cols=32,
        burst_cols=32, spad_row=4096, source_stride_bytes=128)
    assert kt_block.minimum_buffer_bytes == 8096
    assert kt_block.commands[0].rs2.immediate == 128
    assert kt_block.commands[1].rs1.byte_offset == 0
    assert kt_block.commands[2].rs1.byte_offset == 16 * 128
    assert kt_block.commands[1].rs2.immediate == (
        (16 << 48) | (32 << 32) | 4096)


def test_managed_scale_protocol_rejects_wrong_half_and_overlap() -> None:
    profile = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="complete selected managed half"):
        lower_strided_scale_dma(profile, buffer="KT_SCALES", row_bytes=32,
                                rows=2, pitch_bytes=128, destination_byte=8,
                                selector=1, gated=True)
    with pytest.raises(ValueError, match="complete selected managed half"):
        lower_strided_scale_dma(profile, buffer="KT_SCALES", row_bytes=32,
                                rows=2, pitch_bytes=128, destination_byte=0,
                                selector=0, gated=True)
    with pytest.raises(ValueError, match="unsupported"):
        lower_matrix_mvin(profile, buffer="KT_IN", matrix_rows=64,
                          matrix_cols=32, burst_cols=32, spad_row=4096,
                          source_stride_bytes=16)
    with pytest.raises(ValueError, match="overlaps"):
        lower_managed_spad_matmul(
            profile, m=64, n=32, k=64, a_row=0, b_end_row=4224,
            output_row=0, weight_half=0, activation_half=1,
            scale_sink="scale_sink")
    plain = load_profile(PROFILE_DIR / "MxGemminiRocketConfig.json")
    with pytest.raises(ValueError, match=r"MX\+VPU"):
        lower_managed_spad_matmul(
            plain, m=64, n=32, k=64, a_row=0, b_end_row=4224,
            output_row=8192, weight_half=0, activation_half=1,
            scale_sink="scale_sink")


def test_typed_flash_scale_and_matmul_verify_in_python_and_native(tmp_path: Path) -> None:
    profile = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    digest, source = profile_sha256(profile), "a" * 64
    mlir = f'''builtin.module attributes {{
  mx.attention_program_schema = "mx_gemmini.nicolas_flash_attention_program.v1",
  mx.profile_sha256 = "{digest}", mx.source_sha256 = "{source}"
}} {{
  func.func @flash_qk0(%kt_scales: memref<160xi8>, %kt: memref<8192xi8>) {{
    "mx_gemmini.gated_scale_dma"(%kt_scales) {{site_id = "qk0:scales",
      row_bytes = 32 : i32, rows = 2 : i32, pitch_bytes = 128 : i32,
      destination_byte = 0 : i32, selector = 1 : i32,
      profile_sha256 = "{digest}", source_sha256 = "{source}"}}
      : (memref<160xi8>) -> ()
    "mx_gemmini.dma_matrix"(%kt) {{site_id = "qk0:weight",
      matrix_rows = 64 : i32, matrix_cols = 32 : i32,
      burst_cols = 32 : i32, spad_row = 4096 : i32,
      source_stride_bytes = 128 : i32, source_offset_bytes = 0 : i32,
      profile_sha256 = "{digest}", source_sha256 = "{source}"}}
      : (memref<8192xi8>) -> ()
    "mx_gemmini.managed_matmul_spad"() {{site_id = "qk0:matmul",
      m = 64 : i32, n = 32 : i32, k = 64 : i32,
      activation_row = 0 : i32, weight_end_row = 4224 : i32,
      output_row = 8192 : i32, weight_half = 0 : i32,
      activation_half = 1 : i32, scale_sink_buffer = "scale_sink",
      profile_sha256 = "{digest}", source_sha256 = "{source}"}} : () -> ()
    func.return
  }}
}}
'''
    report = verify_ir(mlir, profile)
    assert (report["memory_phases"], report["spad_matmuls"]) == (2, 1)
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.is_file():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    source_path = tmp_path / "flash_qk0.mlir"
    source_path.write_text(mlir)
    valid = subprocess.run([str(executable.resolve()), str(source_path), "-o", "/dev/null"],
                           capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    source_path.write_text(mlir.replace("destination_byte = 0 : i32",
                                        "destination_byte = 8 : i32"))
    invalid = subprocess.run([str(executable.resolve()), str(source_path), "-o", "/dev/null"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "managed half" in invalid.stderr
