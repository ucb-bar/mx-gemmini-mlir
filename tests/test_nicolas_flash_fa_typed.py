"""The default flash shape lowers an explicit VPU scale before each softmax."""

import gzip
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_mid import (FA_SHAPE, flash_buffer_abi,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = ("static uint64_t run_serial(void)\n"
        "gemmini_mx_load_scales_2d_gated\n"
        "gemmini_mxquant_config_mvout_managed\n")


def _inputs() -> tuple[str, str]:
    return ((DATA / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((DATA / "attn_vpu_fa.h.gz").read_bytes()).decode())


def test_default_flash_shape_compiles_four_scaled_blocks(tmp_path: Path) -> None:
    frontend, header = _inputs()
    mlir, binding = render_flash_mid(frontend, CORE, CORE, header, PROFILE,
                                     shape=FA_SHAPE, block_k=64)
    report = verify_ir(mlir, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 43, 4)
    commands = lower_flash_mid(mlir, frontend, CORE, CORE, header, PROFILE,
                               shape=FA_SHAPE, block_k=64)
    assert sum(isinstance(item, Command) for item in commands) == 236
    assert sum(isinstance(item, Fence) for item in commands) == 21
    assert mlir.count('kind = "muls"') == 4
    assert FA_SHAPE.scale_immediate_bf16 == 0x3db5
    assert binding["policy"]["fold_q_scale_shift"] == 0
    buffers, lengths = flash_buffer_abi(FA_SHAPE, 64)
    assert buffers[0] == "Q_SCALES" and lengths[-1] == 16384
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_fa.mlir"
        source.write_text(mlir)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_default_flash_rejects_changed_scale_and_unselected_shape() -> None:
    frontend, header = _inputs()
    mlir, _ = render_flash_mid(frontend, CORE, CORE, header, PROFILE,
                               shape=FA_SHAPE, block_k=64)
    with pytest.raises(ValueError, match="typed graph"):
        lower_flash_mid(mlir.replace("immediate_bf16 = 15797 : i32",
                                     "immediate_bf16 = 0 : i32", 1),
                        frontend, CORE, CORE, header, PROFILE,
                        shape=FA_SHAPE, block_k=64)
    with pytest.raises(ValueError, match="selected source shape"):
        flash_buffer_abi(FA_SHAPE, 32)
