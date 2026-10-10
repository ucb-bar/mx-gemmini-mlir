"""Check that archived large asymmetric Spike evidence remains reproducible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVIDENCE = (Path(__file__).resolve().parents[1] / "docs/evidence" /
            "nicolas_asym_large_direct_266c593")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_large_direct_source_receipts_and_generated_artifacts() -> None:
    manifest = json.loads((EVIDENCE / "qualification.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_asymmetric_large_direct_spike_matrix.v1"
    assert [(row["mesh_dim"], row["shape_mnk"], row["matched_bf16_outputs"])
            for row in manifest["cases"]] == [
                (16, [128, 128, 128], 16384),
                (32, [128, 128, 256], 16384)]
    for row in manifest["cases"]:
        name = f"dim{row['mesh_dim']}"
        first_path = EVIDENCE / f"{name}_first.json"
        repro_path = EVIDENCE / f"{name}_repro.json"
        assert _sha(first_path) == row["first_receipt_sha256"]
        assert _sha(repro_path) == row["repro_receipt_sha256"]
        first = json.loads(first_path.read_text())
        repro = json.loads(repro_path.read_text())
        for receipt in (first, repro):
            assert receipt["status"] == "source_golden_matched_on_pinned_spike"
            assert receipt["compared_bf16_outputs"] == 16384
            assert receipt["capture_sites"][0]["shape"] == row["shape_mnk"]
            assert receipt["compiler_revision"] == manifest["compiler_revision"]
            assert receipt["rtl_revision"] == manifest["rtl_revision"]
            receipt["build_log_sha256"].pop("link.log")
        assert first == repro
        assert row["deterministic_fields_match"] is True
        for artifact, expected in row["artifacts_sha256"].items():
            assert _sha(EVIDENCE / artifact) == expected
