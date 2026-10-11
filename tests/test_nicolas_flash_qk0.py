"""A typed first flash QK block lowers to the checked managed command sequence."""

import gzip
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.nicolas_flash_qk0 import (BUFFERS, lower_flash_qk0,
                                                  render_flash_qk0)
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = "gemmini_mx_load_scales_2d_gated gemmini_mxquant_config_mvout_managed"
WRAPPER = '#define BK 32\n#include "attn_flash.c"\n'


def _inputs() -> tuple[str, str]:
    return ((DATA / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((DATA / "attn_vpu_mid.h.gz").read_bytes()).decode())


def test_flash_qk0_compiles_from_source_bound_model_graph(tmp_path: Path) -> None:
    frontend, header = _inputs()
    mlir, binding = render_flash_qk0(frontend, CORE, WRAPPER, header, PROFILE)
    commands = lower_flash_qk0(mlir, frontend, CORE, WRAPPER, header, PROFILE)
    assert len(commands) == 36
    assert [command.funct for command in commands if hasattr(command, "funct")][
        13:18] == [27, 26, 9, 24, 8]
    assert binding["policy"]["fold_q_scale_shift"] == 3
    assert BUFFERS == ("Q_SCALES_FOLDED", "KT_SCALES", "Q_IN", "KT_IN",
                       "scale_sink", "S0_HW")
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_qk0.mlir"
        source.write_text(mlir)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_flash_qk0_rejects_changed_typed_geometry_and_frontend() -> None:
    frontend, header = _inputs()
    mlir, _ = render_flash_qk0(frontend, CORE, WRAPPER, header, PROFILE)
    with pytest.raises(ValueError, match="typed graph"):
        lower_flash_qk0(mlir.replace("weight_half = 0 : i32",
                                    "weight_half = 1 : i32"),
                        frontend, CORE, WRAPPER, header, PROFILE)
    with pytest.raises(ValueError, match="source or frontend"):
        render_flash_qk0(frontend.replace('prov.region_id = "matmul_0"',
                                          'prov.region_id = "wrong"'),
                          CORE, WRAPPER, header, PROFILE)
