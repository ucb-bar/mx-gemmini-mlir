"""Match the current source drivers' documented feasible and infeasible tiles."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_gemm import (plan_source_gemm, read_source_gemm,
                                            source_scratchpad_bytes)
from mx_gemmini_support.target_profile import load_profile
from tools.materialize_radiance_ws_data import materialize


SOURCE = os.getenv("RADIANCE_KERNELS_ROOT")
pytestmark = pytest.mark.skipif(not SOURCE, reason="set RADIANCE_KERNELS_ROOT for source comparison")


def test_feasible_fp8_replacement_matches_source_layout_and_current_target():
    root = Path(SOURCE)
    directory = root / "kernels/gemm_mxgemmini"
    kernel = read_source_gemm(
        directory / "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp")
    source_bytes = source_scratchpad_bytes(root / "lib/mxgemm/mxgemm_lib.hpp")
    plan = plan_source_gemm(kernel, scratchpad_bytes=source_bytes)
    assert plan["shape"] == [128, 128, 512]
    assert plan["tile"] == [128, 128, 128]
    assert plan["c_spad_dest"] == 3072
    assert len(plan["waves"]) == 4
    assert [wave["a_spad_start"] for wave in plan["waves"]] == [0, 2048, 0, 2048]
    assert [wave["b_spad_end"] for wave in plan["waves"]] == [8192, 6144, 8192, 6144]
    assert [wave["accumulate"] for wave in plan["waves"]] == [False, True, True, True]
    assert [wave["move_acc_to_spad"] for wave in plan["waves"]] == [False, False, False, True]
    profile = load_profile(Path(__file__).resolve().parents[1] /
                           "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    target = plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                              profile=profile)
    assert target["c_spad_dest"] == 1024
    assert target["scratchpad_bytes"] == 262144


def test_infeasible_source_tile_is_rejected_before_command_lowering():
    root = Path(SOURCE)
    directory = root / "kernels/gemm_mxgemmini"
    kernel = read_source_gemm(
        directory / "mxgemm.fp8.m128n128k512.tm128tn128tk256.fullout.cpp")
    source_bytes = source_scratchpad_bytes(root / "lib/mxgemm/mxgemm_lib.hpp")
    with pytest.raises(ValueError, match="C does not fit"):
        plan_source_gemm(kernel, scratchpad_bytes=source_bytes)


def test_fp6_source_needs_a_lut_capable_target():
    root = Path(SOURCE)
    directory = root / "kernels/gemm_mxgemmini"
    kernel = read_source_gemm(
        directory / "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp")
    source_plan = plan_source_gemm(
        kernel, scratchpad_bytes=source_scratchpad_bytes(root / "lib/mxgemm/mxgemm_lib.hpp"))
    assert source_plan["lut_once"]
    assert len(source_plan["waves"]) == 16
    profile = load_profile(Path(__file__).resolve().parents[1] /
                           "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="no fp6_e3m2/lut"):
        plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                         profile=profile)


def test_read_once_weight_stationary_sources_bind_real_data(tmp_path):
    root = Path(SOURCE)
    report = materialize(root)
    assert report["generated"] == []
    profile = load_profile(Path(__file__).resolve().parents[1] /
                           "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    for folder, shape, precision, waves in (
            ("gemm_mxgemmini_ws", (256, 64, 2048), "FP8", 32),
            ("gemm_mxgemmini_ws_downproj_fp4", (256, 64, 5632), "FP4", 88)):
        driver = root / "kernels" / folder / "kernel.cpp"
        kernel = read_source_gemm(driver)
        assert kernel.shape == shape
        assert kernel.tile == (256, 64, 64)
        assert kernel.datatype == precision
        assert kernel.data_header == driver.parent / "data"
        plan = plan_source_gemm(
            kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
            profile=profile)
        assert len(plan["waves"]) == waves
        assert "output_tiles" not in plan
    original = root / "kernels/gemm_mxgemmini_ws/kernel.cpp"
    changed = tmp_path / "gemm_mxgemmini_ws"
    changed.mkdir()
    (changed / "data").symlink_to(original.parent / "data")
    (changed / "kernel.cpp").write_text(
        original.read_text().replace("mxgemm<CFG>", "mxgemm<OTHER>", 1))
    with pytest.raises(ValueError, match="shared GEMM library"):
        read_source_gemm(changed / "kernel.cpp")
