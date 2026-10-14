"""The captured SmolVLA mask remains connected to both flash shapes."""

from pathlib import Path

import pytest

from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_flash_mid import (
    flash_buffer_abi, lower_flash_mid, render_flash_mid,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evidence/smolvla_attention_cutpoint_20261014"
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/"
    "MxE4M3Fp4VpuGemminiRocketConfig.json")
SOURCE = (
    "static uint64_t run_serial(void)\n"
    "gemmini_mx_load_scales_2d_gated\n"
    "gemmini_mxquant_config_mvout_managed\n"
    "gemmini_vpu_binary(VPU_ADD, S, S, SP_MASK + ROWS16(SQ, BOFF(j)), rows);\n"
)


@pytest.mark.parametrize("keys,physical,commands,fences,vpu", [
    (113, 128, 178, 11, 19),
    (163, 192, 248, 16, 31),
])
def test_captured_masked_head_lowers(keys: int, physical: int,
                                    commands: int, fences: int,
                                    vpu: int) -> None:
    frontend = (DATA / f"attention{keys}.model2mlir.mlir").read_text()
    header = (DATA / ("smol_head113.h" if keys == 113 else "smol_head0.h")).read_text()
    shape = DenseAttentionShape(64, physical, 64)
    graph, binding = render_flash_mid(
        frontend, SOURCE, SOURCE, header, PROFILE, shape=shape,
        block_k=64, arbitrary_mask=True)
    report = verify_ir(graph, PROFILE)
    assert report["vpu_commands"] == vpu
    assert binding["policy"]["arbitrary_mask"] is True
    assert graph.count('site_id = "operand:arbitrary_mask"') == 1
    assert graph.count(':mask"') == physical // 64
    names, sizes = flash_buffer_abi(shape, 64, arbitrary_mask=True)
    assert names[-1] == "MASK_BF16" and sizes[-1] == 64 * physical * 2
    lowered = lower_flash_mid(graph, frontend, SOURCE, SOURCE, header,
                              PROFILE, shape=shape, block_k=64,
                              arbitrary_mask=True)
    assert len(lowered) == commands
    assert report["vpu_commands"] + report["memory_phases"] < commands
    assert sum(item.__class__.__name__ == "Fence" for item in lowered) == fences


def test_mask_cannot_disappear_from_frontend_or_source() -> None:
    frontend = (DATA / "attention113.model2mlir.mlir").read_text()
    header = (DATA / "smol_head113.h").read_text()
    shape = DenseAttentionShape(64, 128, 64)
    with pytest.raises(ValueError, match="selected SmolVLA"):
        flash_buffer_abi(shape, 32, arbitrary_mask=True)
    with pytest.raises(ValueError, match="score-mask"):
        render_flash_mid(frontend.replace('prov.region_id = "add_0"',
                                          'prov.region_id = "other"'),
                         SOURCE, SOURCE, header, PROFILE, shape=shape,
                         block_k=64, arbitrary_mask=True)
    with pytest.raises(ValueError, match="source schedule"):
        render_flash_mid(frontend, SOURCE.replace("gemmini_vpu_binary", "missing"),
                         SOURCE.replace("gemmini_vpu_binary", "missing"),
                         header, PROFILE, shape=shape, block_k=64,
                         arbitrary_mask=True)
