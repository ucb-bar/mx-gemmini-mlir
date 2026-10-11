"""Ensure the Nicolas roster report does not promote provenance to parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_nicolas_regeneration import build_report


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json"
AUDIT = ROOT / "docs/evidence/nicolas_mx_regeneration_audit_266c593/index.json"


def test_report_revalidates_every_selected_matrix_receipt() -> None:
    inventory = json.loads(INVENTORY.read_text())
    report = json.loads(AUDIT.read_text())
    assert build_report(inventory) == report
    assert report["programs"] == len(report["entries"]) == 171
    assert report["status_counts"] == {
        "generated_object_selected_spike_result_matched": 156,
        "separate_evidence_requires_scope_review": 13,
        "no_direct_source_receipt": 2,
    }
    rows = {row["name"]: row for row in report["entries"]}
    assert {name for name, row in rows.items()
            if row["status"] == "no_direct_source_receipt"} == {
                "matmul_ws_mx_generic", "matmul_single_tile_test"}
    chains = {name for name, row in rows.items()
              if row["family"] == "other_tiled_matrix" and
              row["status"] == "separate_evidence_requires_scope_review"}
    assert chains == {f"matmul_tiled_{precision}_{shape}_chain"
                      for precision in ("fp4", "fp6", "fp8")
                      for shape in ("64x64", "128x128")}
    for row in rows.values():
        if row["status"] == "generated_object_selected_spike_result_matched":
            assert row["family"] in {"asymmetric_matrix", "other_tiled_matrix"}
            assert row["selected_spike_result"]["checked_output_metrics"]
        else:
            assert "selected_spike_result" not in row


def test_report_rejects_stale_source_binding() -> None:
    inventory = json.loads(INVENTORY.read_text())
    matrix = next(row for row in inventory["entries"] if row["name"] ==
                  "matmul_tiled_fp8_128x128")
    matrix["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale source binding"):
        build_report(inventory)
