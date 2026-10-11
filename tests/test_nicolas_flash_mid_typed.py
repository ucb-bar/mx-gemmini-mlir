"""Whole four-block flash graph has an executable, source-bound serial lowering."""

import gzip
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_commands import lower_flash_commands
from mx_gemmini_support.nicolas_flash_mid import (BUFFERS, BUFFER_BYTES,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = ("static uint64_t run_serial(void)\n"
        "gemmini_mx_load_scales_2d_gated\n"
        "gemmini_mxquant_config_mvout_managed\n")
WRAPPER = '#define BK 32\n#include "attn_flash.c"\n'


def _inputs() -> tuple[str, str]:
    return ((DATA / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((DATA / "attn_vpu_mid.h.gz").read_bytes()).decode())


def test_serial_flash_mid_graph_lowers_all_blocks_and_vector_stages(tmp_path: Path) -> None:
    frontend, header = _inputs()
    mlir, binding = render_flash_mid(frontend, CORE, WRAPPER, header, PROFILE)
    report = verify_ir(mlir, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 39, 4)
    commands = lower_flash_mid(mlir, frontend, CORE, WRAPPER, header, PROFILE)
    assert sum(isinstance(item, Command) for item in commands) == 156
    assert sum(isinstance(item, Fence) for item in commands) == 21
    assert binding["policy"]["score_rows"] == [8192, 12288, 8192, 12288]
    assert binding["policy"]["fenced"] is True
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_mid.mlir"
        source.write_text(mlir)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_flash_mid_requires_gated_weight_slice_before_managed_matmul() -> None:
    frontend, header = _inputs()
    mlir, _ = render_flash_mid(frontend, CORE, WRAPPER, header, PROFILE)
    lines = mlir.splitlines()
    lines = [line for line in lines if 'site_id = "qk0:scales"' not in line]
    with pytest.raises(ValueError, match="lacks its gated"):
        lower_flash_commands("\n".join(lines) + "\n", PROFILE, BUFFERS,
                             BUFFER_BYTES)
    with pytest.raises(ValueError, match="typed graph"):
        lower_flash_mid(mlir.replace("weight_half = 0 : i32",
                                     "weight_half = 1 : i32", 1),
                        frontend, CORE, WRAPPER, header, PROFILE)


