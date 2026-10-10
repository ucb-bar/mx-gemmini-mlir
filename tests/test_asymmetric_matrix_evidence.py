"""Keep the archived Nicolas Spike matrix claims tied to their source receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _cell(cell: dict) -> str:
    return json.dumps(cell, sort_keys=True)


@pytest.mark.parametrize("dim,folder_name,profile_name,mode_count,missing_count,compiler", [
    (8, "nicolas_asym_matrix_dim8_266c593",
     "MxDim8AllAsymGemminiRocketConfig", 21, 15, "de188c1"),
    (16, "nicolas_asym_matrix_dim16_all_266c593",
     "MxAllAsymGemminiRocketConfig", 26, 10, "0e03168"),
    (32, "nicolas_asym_matrix_dim32_266c593",
     "MxDim32AllAsymGemminiRocketConfig", 21, 15, "de188c1"),
])
def test_all_asymmetric_mesh_receipts_cover_checked_in_source_modes(
        dim: int, folder_name: str, profile_name: str, mode_count: int,
        missing_count: int, compiler: str) -> None:
    folder = EVIDENCE / folder_name
    profile = json.loads((PROFILES / f"{profile_name}.json").read_text())
    legal = {_cell(cell) for cell in profile["legal_compute"]}
    manifests = [json.loads((folder / f"matrix_{run}.json").read_text())
                 for run in ("first", "repro")]
    for manifest, run in zip(manifests, ("first", "repro")):
        assert (manifest["mesh_dim"], manifest["selected_modes"],
                manifest["passed_modes"], manifest["selected_profiles"],
                manifest["legal_mode_count"], manifest["profile_complete"]) == (
                    dim, mode_count, mode_count, 1, 36, False)
        if dim == 16:
            assert manifest["all_asym_profile"] is True
        selected = {_cell(row["compute"]) for row in manifest["rows"]}
        assert len(selected) == mode_count
        assert selected <= legal
        assert len(manifest["uncovered_legal_compute"]) == missing_count
        assert {(_cell(item["compute"]), item["profile_name"])
                for item in manifest["uncovered_legal_compute"]} == {
                    (cell, profile_name) for cell in legal - selected}
        for row in manifest["rows"]:
            assert row["status"] == "passed"
            assert row["profile_name"] == profile_name
            assert row["matched_bf16_outputs"] == 4096
            receipt_path = folder / run / f"{row['source_suffix']}.json"
            receipt_bytes = receipt_path.read_bytes()
            assert hashlib.sha256(receipt_bytes).hexdigest() == row["receipt_sha256"]
            receipt = json.loads(receipt_bytes)
            assert receipt["mesh_dim"] == dim
            assert receipt["status"] == "source_golden_matched_on_pinned_spike"
            assert receipt["compared_bf16_outputs"] == 4096
            assert receipt["rtl_revision"].startswith("266c593")
            assert receipt["compiler_revision"].startswith(compiler)

    first, repro = manifests
    assert first["uncovered_legal_compute"] == repro["uncovered_legal_compute"]
    assert [row["source_suffix"] for row in first["rows"]] == [
        row["source_suffix"] for row in repro["rows"]]
    for row in first["rows"]:
        name = row["source_suffix"]
        a = json.loads((folder / "first" / f"{name}.json").read_text())
        b = json.loads((folder / "repro" / f"{name}.json").read_text())
        for receipt in (a, b):
            receipt["build_log_sha256"].pop("link.log")
        assert a == b
