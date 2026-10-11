"""Guard Nicolas's source-bound flat and tiled FP8 SPAD_REQUANT."""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.dual_spad_requant import (FP8_SPEC, lower_dual_requant,
                                                   render_dual_requant)
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_spad_requant_fp8 import _compiler_driver


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"


def test_source_bound_fp8_flat_and_tiled_lowering() -> None:
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/spad_requant.c"
    header = RTL / "software/gemmini-rocc-tests/include/mx_e4m3_ref.h"
    if not source.is_file() or not header.is_file():
        pytest.skip("requires Nicolas's pinned FP8 source and reference header")
    profile = load_profile(PROFILE, rtl_root=RTL)
    graph = render_dual_requant(
        profile, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        spec=FP8_SPEC)
    commands = lower_dual_requant(graph, profile, spec=FP8_SPEC)
    codes = [item for item in commands if isinstance(item, Command)]
    assert Counter(item.funct for item in codes) == {0: 2, 2: 16, 3: 16, 7: 1, 34: 2}
    requants = [item for item in codes if item.funct == 34]
    assert [item.rs1.buffer for item in requants] == ["scales_hw", "scales_hw2"]
    assert all(not item.rs2.immediate & (1 << 32) for item in requants)
    assert [bool(item.rs1.or_bits & (1 << 28)) for item in requants] == [False, True]
    with pytest.raises(ValueError, match="output binding differs"):
        lower_dual_requant(
            graph.replace('scale_buffer = "scales_hw2"',
                          'scale_buffer = "scales_hw"'), profile, spec=FP8_SPEC)
    with pytest.raises(ValueError, match="SSA edges differ"):
        lower_dual_requant(
            graph.replace("func.return %flat, %flats, %tiled, %tileds",
                          "func.return %tiled, %flats, %flat, %tileds"),
            profile, spec=FP8_SPEC)


def test_fp8_driver_preserves_oracle_and_replaces_issue() -> None:
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/spad_requant.c"
    if not source.is_file():
        pytest.skip("requires Nicolas's pinned FP8 source")
    original = source.read_text()
    modified = _compiler_driver(original)
    assert "mxr_quant_block(&X[m][32 * b]" in modified
    assert "spad_requant %s" in modified
    assert "gemmini_spad_requant(" not in modified
    assert "gemmini_extended_mvin(" not in modified
    assert "gemmini_extended_mvout(" not in modified
    assert modified.count("mx_issue(") == 2
    with pytest.raises(ValueError, match="anchor changed"):
        _compiler_driver(original.replace("  gemmini_flush(0);", "  gemmini_flush(1);"))
    with pytest.raises(ValueError, match="geometry or oracle changed"):
        _compiler_driver(original.replace("#define SP_TILED 0x2000",
                                          "#define SP_TILED 0x3000"))
