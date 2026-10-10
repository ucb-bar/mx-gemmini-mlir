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
    assert archived["named_profiles_with_indexed_spike_evidence"] == 10
    assert {row["mesh_dim"] for row in archived["profiles"]} == {8, 16, 32}
    assert all(row["profile_qualification"] == "structural_unqualified"
               for row in archived["profiles"])


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
    assert selected["named_profile_spike_evidence"] == [{
        "kind": "radiance_gemm_roster_spike",
        "evidence": "docs/evidence/radiance_mx_vpu_legal_roster_266c593/index.json",
        "precisions": ["FP4", "FP8"], "cases_per_run": 24, "runs": 2,
        "issues_vpu_commands": False,
    }]
    assert all(not row["named_profile_spike_evidence"] for name, row in vpu.items()
               if name != selected["name"])
