"""Guard the source-bound dual-layout FP4 SPAD_REQUANT lowering."""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.fp4_dual_requant import lower_fp4_dual_requant, render_fp4_dual_requant
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_spad_requant_fp4 import _compiler_driver


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


@pytest.mark.parametrize("profile_name", (
    "MxE4M3Fp4VpuGemminiRocketConfig", "MxE4M3VpuGemminiRocketConfig"))
def test_both_vpu_profiles_accept_typed_fp4_dual_requant(profile_name: str) -> None:
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/spad_requant_fp4.c"
    header = RTL / "software/gemmini-rocc-tests/include/mx_e4m3_ref.h"
    if not source.is_file() or not header.is_file():
        pytest.skip("requires Nicolas's pinned FP4 source and reference header")
    profile = load_profile(PROFILE_DIR / f"{profile_name}.json", rtl_root=RTL)
    graph = render_fp4_dual_requant(
        profile, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
    commands = lower_fp4_dual_requant(graph, profile)
    codes = [item for item in commands if isinstance(item, Command)]
    assert Counter(item.funct for item in codes) == {0: 2, 2: 64, 3: 32, 7: 1, 34: 2}
    requants = [item for item in codes if item.funct == 34]
    assert [item.rs1.buffer for item in requants] == ["scales_hw", "scales_hw2"]
    assert all(item.rs2.immediate & (1 << 32) for item in requants)
    assert bool(requants[0].rs1.or_bits & (1 << 28)) is False
    assert bool(requants[1].rs1.or_bits & (1 << 28)) is True
    altered = graph.replace('scale_buffer = "scales_hw2"',
                            'scale_buffer = "scales_hw"')
    with pytest.raises(ValueError, match="output binding differs"):
        lower_fp4_dual_requant(altered, profile)
    altered = graph.replace("func.return %flat, %flats, %tiled, %tileds",
                            "func.return %tiled, %flats, %flat, %tileds")
    with pytest.raises(ValueError, match="SSA edges differ"):
        lower_fp4_dual_requant(altered, profile)


def test_compiler_driver_keeps_source_oracle_but_removes_issue_calls() -> None:
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/spad_requant_fp4.c"
    if not source.is_file():
        pytest.skip("requires Nicolas's pinned FP4 source")
    original = source.read_text()
    modified = _compiler_driver(original)
    assert "codes_ref[m][32 * b + k] = fp4_code" in modified
    assert "uint8_t sc = mxr_scale" in modified
    assert "gemmini_spad_requant_fp4(" not in modified
    assert "gemmini_extended_mvin(" not in modified
    assert "gemmini_extended_mvout(" not in modified
    assert modified.count("mx_issue(") == 2
    with pytest.raises(ValueError, match="anchor changed"):
        _compiler_driver(original.replace("  gemmini_flush(0);", "  gemmini_flush(1);"))
    with pytest.raises(ValueError, match="geometry or reference changed"):
        _compiler_driver(original.replace("#define SP_TILED 0x2000",
                                          "#define SP_TILED 0x3000"))


def test_plain_mx_profile_rejects_fp4_spad_requant() -> None:
    profile = load_profile(PROFILE_DIR / "MxGemminiRocketConfig.json")
    graph = render_fp4_dual_requant(
        profile, source_sha256="0" * 64, header_sha256="1" * 64)
    with pytest.raises(ValueError, match="no SPAD_REQUANT"):
        lower_fp4_dual_requant(graph, profile)
