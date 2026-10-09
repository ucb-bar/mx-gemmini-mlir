"""Current RTL profile census and fail-closed format/mode selection."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import (
    FORMATS, SCHEMA, _legal_tuples, export_profiles, load_profile, profile_sha256,
    require_compute, verify_profile_source,
)


def test_lut_projection_selects_mode_per_operand():
    modes = set(range(12))
    cells = _legal_tuples({"fp8_e4m3"}, {"fp4_e2m1"}, modes, has_lut=True)
    assert {(cell["activation_projection"], cell["pe_mode"]) for cell in cells} == {
        ("direct", 6), ("lut", 10),
    }
    cells = _legal_tuples({"fp6_e2m3"}, {"fp8_e4m3"}, modes, has_lut=True)
    assert {(cell["weight_projection"], cell["pe_mode"]) for cell in cells} == {
        ("direct", 8), ("lut", 9),
    }
    assert _legal_tuples({"fp6_e2m3"}, {"fp4_e2m1"}, modes, has_lut=False) == []


@pytest.fixture(scope="module")
def profiles():
    root = os.getenv("MX_GEMMINI_RTL_ROOT")
    if not root:
        pytest.skip("set MX_GEMMINI_RTL_ROOT to a Gemmini checkout with the pinned MxGen submodule")
    return export_profiles(Path(root))


def test_latest_named_rtl_census(profiles):
    assert len(profiles) == 81
    assert sum(profile["chipyard_config"] is not None for profile in profiles.values()) == 40
    assert sum(profile["chipyard_config"] is None for profile in profiles.values()) == 41
    assert all(profile["schema"] == SCHEMA for profile in profiles.values())
    assert all(profile["legal_compute"] for profile in profiles.values())
    assert all(len(profile_sha256(profile)) == 64 for profile in profiles.values())
    for profile in profiles.values():
        assert len(profile["legal_compute"]) == len({
            tuple(sorted(cell.items())) for cell in profile["legal_compute"]})
        assert all(cell["activation_format"] in FORMATS and cell["weight_format"] in FORMATS
                   for cell in profile["legal_compute"])


def test_all_asym_contains_all_pairs_and_exact_modes(profiles):
    profile = profiles["MxAllAsymGemminiRocketConfig"]
    assert len(profile["legal_compute"]) == 36
    assert len({(cell["activation_format"], cell["weight_format"])
                for cell in profile["legal_compute"]}) == 25
    require_compute(profile, "fp8_e4m3", "fp4_e2m1", pe_mode=10,
                    activation_projection="lut", weight_projection="direct")
    with pytest.raises(ValueError, match="absent"):
        require_compute(profile, "fp8_e4m3", "fp4_e2m1", pe_mode=11,
                        activation_projection="lut", weight_projection="direct")


def test_single_format_and_dim_profiles(profiles):
    single = profiles["MxE4M3SingleGemminiRocketConfig"]
    assert single["resources"]["lut"] is False
    assert single["legal_compute"] == [{
        "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
        "activation_projection": "direct", "weight_projection": "direct", "pe_mode": 8,
    }]
    assert profiles["MxDim8AllAsymGemminiRocketConfig"]["geometry"]["mesh_rows"] == 8
    assert profiles["MxDim32AllAsymGemminiRocketConfig"]["geometry"]["mesh_rows"] == 32
    assert profiles["MxE4M3Fp4VpuGemminiRocketConfig"]["resources"]["vpu"] is True
    assert {cell["pe_mode"] for cell in profiles["MxE4M3Fp4VpuGemminiRocketConfig"]["legal_compute"]} == {0, 8}
    assert profiles["MxDim32AllAsymGemminiRocketConfig"]["resources"]["scale_mem_config"]["subbank_line_bytes"] == 32
    assert profiles["MxE4M3Fp4VpuGemminiRocketConfig"]["resources"]["vpu_config"] == {
        "units": 2, "exp_sub": True, "exp_sum": True}
    assert profiles["MxE4M3Fp4VpuGemminiRocketConfig"]["chipyard_overrides"] == {
        "system_bus_width_bits": 512, "rocket_cores": 1, "l2_banks": 4,
        "l2_outer_latency_cycles": 128, "serial_tl_sink_bits": 16}


def test_checked_in_census_matches_selected_sources(profiles):
    root = Path(os.environ["MX_GEMMINI_RTL_ROOT"])
    output = Path(__file__).resolve().parents[1] / "profiles/gemmini-mx-cleanup-266c593"
    assert {path.stem for path in output.glob("*.json")} == set(profiles)
    for name, expected in profiles.items():
        assert load_profile(output / f"{name}.json", rtl_root=root) == expected


def test_profile_source_binding(profiles, tmp_path):
    root = Path(os.environ["MX_GEMMINI_RTL_ROOT"])
    profile = profiles["MxAllAsymGemminiRocketConfig"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert load_profile(path, rtl_root=root) == profile
    changed = json.loads(path.read_text())
    changed["source"]["sha256"]["mxgen/src/main/scala/mxgen/Classifier.scala"] = "0" * 64
    with pytest.raises(ValueError, match="source closure"):
        verify_profile_source(changed, root)
    changed = json.loads(path.read_text())
    changed["legal_compute"].append({
        "activation_format": "fp4_e2m1", "weight_format": "fp4_e2m1",
        "activation_projection": "lut", "weight_projection": "direct", "pe_mode": 0,
    })
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="capabilities differ"):
        load_profile(path, rtl_root=root)
