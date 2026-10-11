"""Check the source lifetime behind Nicolas's 64³ resident row reuse."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.resident_pair_plan import plan_fp8_resident_pair
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def test_direct_64_chain_reuses_consumed_a_rows_only() -> None:
    profile = load_profile(PROFILE)
    selected = dict(shape=(64, 64, 64), a_row=0, c1_row=128, c2_row=512)
    with pytest.raises(ValueError, match="overlap"):
        plan_fp8_resident_pair(profile, **selected)
    plan = plan_fp8_resident_pair(profile, **selected, allow_a_c1_reuse=True)
    assert (plan.a_row, plan.c1_row, plan.c2_row) == (0, 128, 512)
    assert plan.a_rows == plan.c_rows == 256
    assert plan.b_row == plan.rows - plan.b_rows

    with pytest.raises(ValueError, match="qualified only"):
        plan_fp8_resident_pair(profile, shape=(128, 128, 128), a_row=0,
                               c1_row=128, c2_row=512,
                               allow_a_c1_reuse=True)
