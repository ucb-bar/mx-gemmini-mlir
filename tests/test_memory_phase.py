"""Profile-checked transfer phases used by Nicolas's MX memory benchmark."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.memory_phase import (
    lower_linear_spad_mvout, lower_matrix_mvin, lower_nicolas_memory_mlir,
    lower_scale_load, render_nicolas_memory_mlir,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


PROFILE = Path(__file__).resolve().parents[1] / (
    "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
SOURCE_SHA = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"


def test_nicolas_memory_phase_command_geometry() -> None:
    profile = load_profile(PROFILE)
    a = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=0)
    b = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=16, spad_row=1024)
    scales = lower_scale_load(profile, buffer="src", payload_bytes=512,
                              operand="activation")
    output = lower_linear_spad_mvout(profile, buffer="dst",
                                     total_bytes=16384, tile_cols=16,
                                     spad_row=0)
    assert [(len(p.commands), p.transferred_bytes, p.row_requests,
             p.minimum_buffer_bytes) for p in (a, b, scales, output)] == [
        (17, 16384, 256, 16384), (65, 16384, 1024, 16384),
        (1, 512, 64, 512), (65, 16384, 1024, 16384)]
    assert [(c.rs1.byte_offset, c.rs2.immediate & 0xffffffff)
            for c in a.commands[1:]] == [
        (i * 16 * 128 + k * 16, (i * 8 + k) * 16)
        for i in range(8) for k in (0, 4)]
    assert [(c.rs1.byte_offset, c.rs2.immediate & 0xffffffff)
            for c in b.commands[1:]] == [
        (i * 16 * 128 + k * 16, 1024 + (i * 8 + k) * 16)
        for i in range(8) for k in range(8)]
    assert [c.rs1.byte_offset for c in output.commands[1:]] == list(
        range(0, 16384, 256))
    assert [c.rs2.immediate & 0xffffffff for c in output.commands[1:]] == list(
        range(0, 1024, 16))
    assert scales.commands[0].funct == 27
    assert scales.commands[0].rs2.immediate == 512


def test_memory_phases_reject_unsupported_profile_or_placement() -> None:
    profile = load_profile(PROFILE)
    small = deepcopy(profile)
    small["resources"]["dma_max_bytes"] = 16
    with pytest.raises(ValueError, match="unsupported"):
        lower_matrix_mvin(small, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=0)
    with pytest.raises(ValueError, match="unsupported"):
        lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=16000)
    with pytest.raises(ValueError, match="exceeds"):
        lower_linear_spad_mvout(profile, buffer="dst", total_bytes=16384,
                                tile_cols=16, spad_row=16000)
    with pytest.raises(ValueError, match="cannot hold"):
        lower_scale_load(profile, buffer="src", payload_bytes=16384,
                         operand="activation")


def test_typed_memory_phases_lower_and_reject_invalid_source_or_dma() -> None:
    profile = load_profile(PROFILE)
    mlir = render_nicolas_memory_mlir(profile, SOURCE_SHA)
    assert verify_ir(mlir, profile)["memory_phases"] == 5
    commands = lower_nicolas_memory_mlir(mlir, profile, SOURCE_SHA)
    assert {name: len(stream) for name, stream in commands.items()} == {
        "setup": 2, "a64": 17, "b16": 65, "scale": 1, "mvout": 65}
    with pytest.raises(ValueError, match="source or target memory binding"):
        verify_ir(mlir.replace(f'source_sha256 = "{SOURCE_SHA}"',
                               'source_sha256 = "' + "0" * 64 + '"', 1), profile)
    with pytest.raises(ValueError, match="unsupported"):
        lower_nicolas_memory_mlir(mlir.replace("burst_cols = 64 : i32",
                                               "burst_cols = 128 : i32"),
                                  profile, SOURCE_SHA)
    with pytest.raises(ValueError, match="too small"):
        verify_ir(mlir.replace("memref<16384xi8>", "memref<1024xi8>"), profile)


def test_native_verifier_checks_typed_memory_buffer_and_source(tmp_path: Path) -> None:
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.is_file():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    mlir = render_nicolas_memory_mlir(load_profile(PROFILE), SOURCE_SHA)
    module = tmp_path / "memory.mlir"
    module.write_text(mlir)
    assert subprocess.run([str(executable.resolve()), str(module), "-o", "/dev/null"],
                          capture_output=True, text=True).returncode == 0
    module.write_text(mlir.replace("memref<16384xi8>", "memref<1024xi8>"))
    invalid = subprocess.run([str(executable.resolve()), str(module), "-o", "/dev/null"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "sufficient size" in invalid.stderr
    module.write_text(mlir.replace(f'mx.source_sha256 = "{SOURCE_SHA}"',
                                   'mx.source_sha256 = "' + "0" * 64 + '"'))
    invalid = subprocess.run([str(executable.resolve()), str(module), "-o", "/dev/null"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "matching source and profile digests" in invalid.stderr
