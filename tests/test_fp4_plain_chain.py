"""Guard packed FP4 SSA edges, placement, and source-bound command lowering."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.fp4_plain_chain import (lower_fp4_plain_chain,
                                                render_fp4_plain_chain,
                                                source_resources)
from mx_gemmini_support.resident_pair_graph import INPUTS, lower_connected_pair
from mx_gemmini_support.resident_pair_plan import plan_fp4_resident_pair
from mx_gemmini_support.target_profile import load_profile
from tools.emit_resident_pair_object import _buffer_abi
from tools.qualify_nicolas_fp4_resident_chain import compiler_driver


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def fixture(dimension: int = 64):
    source_path = (RTL / "software/gemmini-rocc-tests/bareMetalC" /
                   f"matmul_tiled_fp4_{dimension}x{dimension}_chain.c")
    header_path = (RTL / "software/gemmini-rocc-tests/include" /
                   f"matmul_fp4_{dimension}x{dimension}_chain.h")
    if not source_path.is_file() or not header_path.is_file():
        pytest.skip("requires Nicolas's pinned FP4 resident source")
    profile = load_profile(PROFILE, rtl_root=RTL)
    source, header = source_path.read_text(), header_path.read_text()
    resources = source_resources(source, header, dimension=dimension)
    mlir = render_fp4_plain_chain(
        profile, resources,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header_path.read_bytes()).hexdigest(),
        dimension=dimension)
    return profile, resources, mlir, source


@pytest.mark.parametrize("dimension", (64, 128))
def test_fp4_source_bound_connected_pair(dimension: int):
    profile, resources, mlir, source = fixture(dimension)
    commands = lower_fp4_plain_chain(mlir, profile, resources, dimension=dimension)
    plan = plan_fp4_resident_pair(profile, shape=(dimension,) * 3,
                                  a_row=0, c1_row=2048, c2_row=4096)
    rows = dimension * dimension // 32
    assert (plan.a_rows, plan.b_rows, plan.c_rows, plan.b_row) == (
        rows, rows, rows, 16384 - rows)
    issued = [command for command in commands if isinstance(command, Command)]
    assert [command.funct for command in issued].count(8) == 2
    assert [command.funct for command in issued].count(26) == 2
    assert [command.funct for command in issued].count(27) == 3
    assert issued[1].rs1.immediate == ((1 << 16) | (2 << 14) | (2 << 12) |
                                      (2 << 10) | (1 << 2))
    assert issued[-1].funct == 3
    pair = lower_connected_pair(
        mlir, profile, resources, buffers={name: name for name in INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled", precision="fp4_e2m1")
    abi = _buffer_abi(pair, {name: name for name in INPUTS},
                      {name: name for name in
                       ("c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled")},
                      precision="fp4_e2m1")
    packed, scales = dimension * dimension // 2, dimension * dimension // 32
    assert {entry["slot"]: entry["minimum_bytes"] for entry in abi} == {
        "a1_activation": packed, "a1_scales": scales,
        "b1_weight": packed, "b1_scales": scales,
        "b2_weight": packed, "b2_scales": scales,
        "c1_scales": scales, "c1_tiled_observed": packed,
        "c2_scales": scales, "c2_tiled": packed}
    driver = compiler_driver(source, dimension=dimension)
    assert driver.count("mx_issue(") == 2  # declaration and call
    assert driver.count('check_nibbles("C1"') == 1
    assert driver.count('check_nibbles("C2"') == 1


@pytest.mark.parametrize("dimension", (64, 128))
def test_fp4_connected_pair_rejects_tampering(dimension: int):
    profile, resources, mlir, _ = fixture(dimension)
    with pytest.raises(ValueError):
        lower_fp4_plain_chain(mlir.replace("activation_row = 2048", "activation_row = 4096"),
                              profile, resources, dimension=dimension)
    with pytest.raises(Exception):
        wrong = mlir.replace(f"tensor<{dimension // 2}x{dimension}xi8>",
                             f"tensor<{dimension}x{dimension}xi8>", 1)
        lower_fp4_plain_chain(wrong, profile, resources, dimension=dimension)
    with pytest.raises(ValueError):
        lower_fp4_plain_chain(mlir, profile,
                              {**resources, "b2_weight": resources["b2_weight"][:-1]},
                              dimension=dimension)
    with pytest.raises(ValueError):
        lower_connected_pair(
            mlir, profile, resources,
            buffers={name: name for name in INPUTS},
            c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
            c2_tiled="c2_tiled", precision="fp8_e4m3")
