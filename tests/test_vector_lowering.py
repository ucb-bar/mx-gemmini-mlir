"""Typed physical MX vector ops lower to the source-checked Rocket issuer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_c, lower_vector_commands


PROFILES = Path(__file__).resolve().parents[1] / "profiles/gemmini-mx-cleanup-266c593"


def _profile(name):
    return load_profile(PROFILES / f"{name}.json")


def _vector_ir(profile):
    digest = profile_sha256(profile)
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    binding = (f'profile_sha256 = "{digest}", contract_sha256 = "{a}", '
               f'policy_sha256 = "{b}", manifest_sha256 = "{c}"')
    return f'''module attributes {{mx.contract_sha256 = "{a}", mx.policy_sha256 = "{b}",
  prov.quantization_manifest_sha256 = "{c}", mx.profile_sha256 = "{digest}"}} {{
  func.func @vector_commands() {{
    "mx_gemmini.vpu_execute"() {{site_id = "softmax", kind = "expsum",
      src1_row = 2048 : i32, src2_row = 4096 : i32, dst_row = 6144 : i32,
      rows = 128 : i32, reduction_length = 8 : i32, broadcast = true,
      immediate_bf16 = 0 : i32, second_dst_row = 7168 : i32, {binding}}} : () -> ()
    "mx_gemmini.spad_requant"() {{site_id = "p-feed", source_row = 4096 : i32,
      destination_row = 3072 : i32, m = 64 : i32, n = 128 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 268435456 : i64, {binding}}} : () -> ()
    func.return
  }}
}}
'''


def test_vpu_mlir_lowers_to_funct_33_34_and_compiles(tmp_path):
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    ir = _vector_ir(profile)
    commands = lower_vector_commands(ir, profile)
    assert [command.funct for command in commands] == [33, 34]
    source = lower_vector_c(ir, profile)
    assert ".insn r 0x7b, 3, 33" in source
    assert ".insn r 0x7b, 3, 34" in source
    path = tmp_path / "vector.c"
    path.write_text(source)
    subprocess.run(["cc", "-fsyntax-only", str(path)], check=True)
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.exists():
        mlir = tmp_path / "vector.mlir"
        mlir.write_text(ir)
        subprocess.run([str(executable.resolve()), str(mlir), "-o", "/dev/null"], check=True)


def test_vpu_mlir_refuses_non_vpu_profile():
    baseline = _profile("MxGemminiRocketConfig")
    with pytest.raises(ValueError, match="no VPU"):
        lower_vector_commands(_vector_ir(baseline), baseline)


def test_vector_mlir_refuses_illegal_row_count(tmp_path):
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    ir = _vector_ir(profile).replace("rows = 128 : i32", "rows = 0 : i32")
    with pytest.raises(ValueError, match="nonzero"):
        lower_vector_commands(ir, profile)
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.exists():
        path = tmp_path / "bad.mlir"
        path.write_text(ir)
        result = subprocess.run([str(executable.resolve()), str(path), "-o", "/dev/null"],
                                text=True, capture_output=True)
        assert result.returncode != 0
        assert "VPU address, row count" in result.stderr


def test_spad_requant_mlir_binds_runtime_scale_buffer(tmp_path):
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    ir = _vector_ir(profile).replace("scale_dram_address = 268435456 : i64",
                                     'scale_dram_address = 0 : i64, scale_buffer = "c1_scales"')
    commands = lower_vector_commands(ir, profile)
    assert commands[-1].rs1.buffer == "c1_scales"
    source = lower_vector_c(ir, profile)
    assert "const void *c1_scales" in source
    path = tmp_path / "buffered.c"
    path.write_text(source)
    subprocess.run(["cc", "-fsyntax-only", str(path)], check=True)
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.exists():
        mlir = tmp_path / "buffered.mlir"
        mlir.write_text(ir)
        subprocess.run([str(executable.resolve()), str(mlir), "-o", "/dev/null"], check=True)
