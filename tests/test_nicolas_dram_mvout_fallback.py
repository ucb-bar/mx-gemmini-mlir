"""Keep Nicolas's Spike fallback selection tied to exact pinned source files."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.smem_readback import _check_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_plain_matrix_object import CASES


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
BASE = ROOT / "docs/evidence/nicolas_fp8_smem_zero_public_64ecacd_266c593/bundle"


@pytest.mark.parametrize("key", (
    "fp8_64x64x64_dram_mvout_spike",
    "fp4_64x64x64_dram_mvout_spike",
    "fp8_128x128x256_dram_mvout_spike",
))
def test_spike_fallback_is_source_and_shape_bound(key: str) -> None:
    profile = load_profile(PROFILE)
    manifest, _ = load_bundle(BASE)
    case = CASES[key]
    selected = {**manifest, "source_driver_sha256": case.source_sha256,
                "precision": case.precision, "shape_mnk": list(case.shape),
                "tile_mnk": list(case.tile)}
    label, shape, source_sha = _check_source(profile, selected)
    assert label.endswith("_spike")
    assert shape == case.shape and source_sha == case.source_sha256

    with pytest.raises(ValueError, match="source/profile"):
        _check_source(profile, {**selected, "precision": "FP6"})
    with pytest.raises(ValueError, match="source/profile"):
        _check_source(profile, {**selected, "tile_mnk": [32, 32, 32]})
    with pytest.raises(ValueError, match="not supported"):
        _check_source(profile, {**selected, "source_driver_sha256": "0" * 64})
