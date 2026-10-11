"""Selected RTL profiles must reach Merlin without invented compute modes."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_profile_contract import contract_for_profile, render_contract
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def test_selected_nicolas_contract_is_current_and_narrow() -> None:
    profile = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    contract = contract_for_profile(profile)
    assert render_contract(profile) == (ROOT / "contracts/target_contract.yaml").read_text()
    assert contract["source"]["rtl_commit"] == profile["source"]["gemmini_revision"]
    assert contract["source"]["rtl_config"] == profile["name"]
    assert contract["certification"]["phase0_admitted"] is False
    assert [unit["name"] for unit in contract["compute_units"]] == ["mx_pe", "vpu"]
    assert contract["compute_units"][0]["accumulate"] == [
        {"in": "mxfp4", "weight": "mxfp4", "acc": "bf16"},
        {"in": "mxfp8", "weight": "mxfp8", "acc": "bf16"},
    ]
    assert "mxfp6" not in contract["compute_units"][0]["dtypes"]
    assert contract["mx_resources"]["scratchpad_bytes"] == 262144
    assert contract["mx_resources"]["vpu_config"]["units"] == 2


def test_other_direct_profile_does_not_inherit_vpu() -> None:
    profile = load_profile(PROFILES / "MxE4M3OnlyGemminiRocketConfig.json")
    contract = contract_for_profile(profile)
    assert [unit["name"] for unit in contract["compute_units"]] == ["mx_pe"]
    assert contract["compute_units"][0]["accumulate"] == [
        {"in": "mxfp8", "weight": "mxfp8", "acc": "bf16"}]

    duplicate = load_profile(PROFILES / "MxE4M3OnlyGemminiRocketConfig.json")
    assert contract_for_profile(duplicate)["compute_units"][0]["accumulate"] == [
        {"in": "mxfp8", "weight": "mxfp8", "acc": "bf16"}]


def test_unsupported_or_changed_profile_fails_closed() -> None:
    profile = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    changed = copy.deepcopy(profile)
    changed["transport"] = "standalone_mmio"
    with pytest.raises(ValueError, match="Rocket RoCC"):
        contract_for_profile(changed)
    changed = copy.deepcopy(profile)
    changed["legal_compute"][0]["activation_projection"] = "lut"
    contract = contract_for_profile(changed)
    assert contract["compute_units"][0]["accumulate"] == [
        {"in": "mxfp8", "weight": "mxfp8", "acc": "bf16"}]
    changed = copy.deepcopy(profile)
    changed["resources"]["scale_block_elements"] = 16
    with pytest.raises(ValueError, match="E8M0"):
        contract_for_profile(changed)
