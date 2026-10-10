"""Audit the original 11-case direct Nicolas matrix object replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_plain_matrix_object import CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION
from tools.qualify_nicolas_plain_matrix_suite import SCHEMA


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_direct_matrix_suite_9f3a759_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_matrix_suite_replays_every_selected_source_case() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    assert index["schema"] == SCHEMA
    assert index["status"] == "all_selected_source_goldens_matched_on_pinned_spike"
    assert index["compiler_revision"] == "9f3a7598704d4fde6b79e93a865fe015d0bbb55f"
    assert index["rtl_revision"] == RTL_REVISION
    assert index["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert index["mxq_revision"] == MXQ_REVISION
    assert len(index["selected_cases"]) == 11
    assert len(index["selected_cases"]) == len(set(index["selected_cases"]))
    assert set(index["selected_cases"]).issubset(CASES)
    assert [row["case"] for row in index["cases"]] == index["selected_cases"]
    assert index["total_outputs_checked"] == sum(
        row["outputs_checked"] for row in index["cases"]) == 111872

    archived_receipts = []
    for path in (ROOT / "docs/evidence").rglob("receipt.json"):
        if path.is_relative_to(ARCHIVE):
            continue
        try:
            record = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if record.get("status") == "source_golden_matched_on_pinned_spike":
            archived_receipts.append((path, record))

    for row in index["cases"]:
        case = CASES[row["case"]]
        path = ARCHIVE / row["receipt"]
        receipt = json.loads(path.read_text())
        assert row["receipt_sha256"] == _sha(path)
        assert row["precision"] == case.precision
        assert row["shape_mnk"] == list(case.shape)
        assert row["quant_output"] == case.quant_output
        assert row["profile_name"] == case.profile_name
        assert row["source_driver_sha256"] == case.source_sha256
        assert row["source_header_sha256"] == case.header_sha256
        assert row["profile_sha256"] == receipt["profile_sha256"]
        assert row["mesh_dim"] == receipt["mesh_dim"]
        assert row["outputs_checked"] == receipt["outputs_checked"]
        assert row["object_sha256"] == receipt["object_sha256"]
        assert row["elf_sha256"] == receipt["elf_sha256"]
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"]
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["mismatches"] == 0
        assert receipt["compiler_revision"] == index["compiler_revision"]

        matches = [prior_path for prior_path, prior in archived_receipts
                   if prior.get("source_driver_sha256") == case.source_sha256
                   and prior.get("object_sha256") == row["object_sha256"]
                   and prior.get("spike_log_sha256") == row["spike_log_sha256"]]
        assert matches, f"{case.key} has no archived full-output artifact"
        assert any((prior.parent / "object/mx_issue.o").is_file() and
                   _sha(prior.parent / "object/mx_issue.o") == row["object_sha256"] and
                   (prior.parent / "run/spike.log").is_file() and
                   _sha(prior.parent / "run/spike.log") == row["spike_log_sha256"]
                   for prior in matches)
