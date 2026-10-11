"""Check the archived current-model2MLIR connected source qualifications."""

from __future__ import annotations

import json
from pathlib import Path

from tools.archive_nicolas_portable_chains import CASES, verify


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_portable_chains_a042643_266c593"


def test_portable_chain_archive_covers_six_full_output_comparisons() -> None:
    manifest = verify(EVIDENCE)
    suite = json.loads((EVIDENCE / "index.json").read_text())
    assert manifest["suite_sha256"] == manifest["files"]["index.json"]["sha256"]
    assert suite["compiler_revision"] == "3d8b8d201d1bcaf3db712ab8093d0f6b5bd6d726"
    assert suite["model2mlir_revision"] == "a042643e31366724ca0482389c4ee85a9f88e343"
    assert suite["model2mlir_source_closure_sha256"] == (
        "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39")
    assert suite["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert [row["case"] for row in suite["rows"]] == list(CASES)
    outputs = [json.loads((EVIDENCE / case / "receipt.json").read_text())["checked_outputs"]
               for case in CASES]
    assert sum(row["c1_codes"] + row["c2_codes"] for row in outputs) == 122880
    assert sum(row["c1_scales"] + row["c2_scales"] for row in outputs) == 3840
