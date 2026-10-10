"""Keep profile capabilities and simulator evidence at their actual scopes."""

from __future__ import annotations

import json
from pathlib import Path

from tools.report_mx_profile_qualification import build_report


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/evidence/mx_profile_qualification_catalog_266c593/index.json"


def test_catalog_rederives_from_pinned_profiles_and_receipts() -> None:
    archived = json.loads(CATALOG.read_text())
    assert archived == build_report()
    assert archived["profile_count"] == 81
    assert archived["named_profiles_with_indexed_spike_evidence"] == 40
    assert archived["chipyard_wrapper_count"] == 40
    assert archived["chipyard_wrappers_with_indexed_spike_evidence"] == 40
    assert {row["mesh_dim"] for row in archived["profiles"]} == {8, 16, 32}
    assert all(row["profile_qualification"] == "structural_unqualified"
               for row in archived["profiles"])
    assert all(row["named_profile_spike_evidence"] for row in archived["profiles"]
               if row["chipyard_wrapper"])
    assert all(not row["named_profile_spike_evidence"] for row in archived["profiles"]
               if not row["chipyard_wrapper"])


def test_stock_failure_and_candidate_patch_stay_separate() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    failed = [row for row in profiles if row["mode_class_stock_spike_failures"]]
    assert len(failed) == 16
    for row in profiles:
        assert (row["mode_class_stock_spike_passes"] +
                len(row["mode_class_stock_spike_failures"]) ==
                row["legal_mode_count"] == row["mode_class_candidate_spike_passes"])
        for cell in row["mode_class_stock_spike_failures"]:
            assert cell == {
                "activation_format": "fp8_e4m3", "activation_projection": "direct",
                "weight_format": "fp8_e4m3", "weight_projection": "lut", "pe_mode": 8,
            }
    vpu = {row["name"]: row for row in profiles if row["vpu"]}
    assert len(vpu) == 4
    selected = vpu["MxE4M3Fp4VpuGemminiRocketConfig"]
    assert selected["legal_mode_count"] == 2
    assert selected["mode_class_stock_spike_passes"] == 2
    assert selected["named_profile_spike_evidence"][0] == {
        "kind": "radiance_gemm_roster_spike",
        "evidence": "docs/evidence/radiance_mx_vpu_legal_roster_266c593/index.json",
        "precisions": ["FP4", "FP8"], "cases_per_run": 24, "runs": 2,
        "issues_vpu_commands": False,
    }
    assert selected["named_profile_spike_evidence"][1]["kind"] == (
        "connected_mx_vpu_narrow_spike")
    base = vpu["MxE4M3VpuGemminiRocketConfig"]
    assert len(base["named_profile_spike_evidence"]) == 7
    assert sum(receipt["cases"] for receipt in
               base["named_profile_spike_evidence"] if "cases" in receipt) == 28
    assert base["named_profile_spike_evidence"][-3]["kind"] == (
        "connected_mx_vpu_narrow_spike")
    assert base["named_profile_spike_evidence"][-2]["kind"] == (
        "connected_mx_vpu_two_tile_spike")
    assert base["named_profile_spike_evidence"][-1]["kind"] == (
        "connected_mx_vpu_two_tile_pipelined_spike")
    assert all(not row["named_profile_spike_evidence"] for name, row in vpu.items()
               if name not in {selected["name"], base["name"]})


def test_both_named_vpu_rocket_profiles_have_connected_output_receipts() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    vpu = {row["name"]: row for row in profiles if row["vpu"]}
    expected = {"c1_bf16_values": 4096, "c1_fp8_codes": 4096,
                "c1_e8m0_scales": 128, "c2_fp8_codes": 2048,
                "c2_e8m0_scales": 64}
    for name in ("MxE4M3Fp4VpuGemminiRocketConfig",
                 "MxE4M3VpuGemminiRocketConfig"):
        receipts = [item for item in vpu[name]["named_profile_spike_evidence"]
                    if item["kind"] == "connected_mx_vpu_narrow_spike"]
        assert len(receipts) == 1
        assert receipts[0]["issues_vpu_commands"] is True
        assert receipts[0]["first_shape_mnk"] == [64, 64, 64]
        assert receipts[0]["second_shape_mnk"] == [64, 32, 64]
        assert receipts[0]["compared"] == expected


def test_both_named_vpu_rocket_profiles_have_full_two_tile_receipts() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    by_name = {row["name"]: row for row in profiles}
    for name in ("MxE4M3Fp4VpuGemminiRocketConfig",
                 "MxE4M3VpuGemminiRocketConfig"):
        receipts = [item for item in by_name[name]["named_profile_spike_evidence"]
                    if item["kind"] == "connected_mx_vpu_two_tile_spike"]
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt["first_shape_mnk"] == [64, 64, 64]
        assert receipt["second_shape_mnk"] == [64, 64, 64]
        assert receipt["second_tile_count"] == 2
        assert receipt["compared_c1_bf16_values"] == 4096
        assert receipt["compared_fp8_codes"] == 16384
        assert receipt["compared_e8m0_scales"] == 512
        assert receipt["issues_vpu_commands"] is True
        assert receipt["issues_captured_mm1"] is True
        assert receipt["schedule"] == "program_order_with_dependency_fences"
        pipelined = [item for item in by_name[name]["named_profile_spike_evidence"]
                     if item["kind"] == "connected_mx_vpu_two_tile_pipelined_spike"]
        assert len(pipelined) == 1
        assert pipelined[0]["schedule"] == "pipelined"
        for key in ("compared_c1_bf16_values", "compared_fp8_codes",
                    "compared_e8m0_scales", "issues_vpu_commands",
                    "issues_captured_mm1"):
            assert pipelined[0][key] == receipt[key]


def test_dedicated_asymmetric_profiles_have_direct_mode_receipts() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    dedicated = [row for row in profiles if row["name"].startswith("MxAsym")]
    assert len(dedicated) == 20
    assert sum(row["legal_mode_count"] for row in dedicated) == 26
    for row in dedicated:
        assert row["named_profile_spike_evidence"] == [{
            "kind": "complete_dedicated_asymmetric_spike_matrix",
            "evidence": "docs/evidence/nicolas_asym_matrix_dim16_266c593/matrix_first.json",
            "passing_modes": row["legal_mode_count"],
            "compared_bf16_outputs": 4096 * row["legal_mode_count"],
        }]


def test_eight_other_wrappers_are_bound_to_their_source_modes() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    selected = [row for row in profiles if any(
        receipt["kind"] == "source_mode_spike_reproduced"
        for receipt in row["named_profile_spike_evidence"])]
    assert len(selected) == 8
    for row in selected:
        assert row["chipyard_wrapper"]
        receipt = row["named_profile_spike_evidence"][0]
        assert receipt["evidence"] == (
            "docs/evidence/nicolas_rocket_wrapper_matrix_266c593/index.json")
        assert (receipt["runs"], receipt["compared_bf16_outputs_per_run"]) == (
            2, 4096)


def test_requantizer_wrapper_has_separate_hardware_output_evidence() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    selected = next(row for row in profiles if
                    row["name"] == "TestRequantizerLutMxGemminiRocketConfig")
    receipt = selected["named_profile_spike_evidence"][-1]
    assert receipt == {
        "kind": "hardware_requantized_output_spike",
        "evidence": "docs/evidence/nicolas_requantizer_wrapper_266c593/index.json",
        "output_precisions": ["fp8", "fp4", "fp6"],
        "cases": 3,
        "fp6_terminal_readout_derived_from_fullout": True,
    }


def test_plain_mx_resident_mm2_receipt_stays_scoped_to_second_contraction() -> None:
    profiles = json.loads(CATALOG.read_text())["profiles"]
    selected = next(row for row in profiles if row["name"] == "MxGemminiRocketConfig")
    receipt = next(item for item in selected["named_profile_spike_evidence"]
                   if item["kind"] == "source_bound_resident_mm2_spike")
    assert receipt == {
        "kind": "source_bound_resident_mm2_spike",
        "evidence": "docs/evidence/nicolas_resident_mm2_128_266c593/index.json",
        "compared_fp8_codes": 16384,
        "compared_e8m0_scales": 512,
        "excludes_mm1": True,
    }
    connected = next(item for item in selected["named_profile_spike_evidence"]
                     if item["kind"] == "source_bound_connected_mm1_mm2_spike")
    assert connected == {
        "kind": "source_bound_connected_mm1_mm2_spike",
        "evidence": "docs/evidence/nicolas_connected_plain_chain_128_266c593/index.json",
        "compared_c1_fp8_codes": 16384,
        "compared_c1_e8m0_scales": 512,
        "compared_c2_fp8_codes": 16384,
        "compared_c2_e8m0_scales": 512,
    }
