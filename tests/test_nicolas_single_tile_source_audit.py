"""Keep the debug-only source evidence distinct from executed compiler results."""

import json
from pathlib import Path

from tools.audit_nicolas_single_tile_debug import audit


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_single_tile_source_audit_266c593/index.json"
RTL = Path("/scratch/agustin/tmp/gemmini-mx-cleanup-20261009")


def test_archived_single_tile_audit_has_no_numerical_oracle() -> None:
    receipt = json.loads(EVIDENCE.read_text())
    assert receipt["status"] == "debug_source_has_no_active_numerical_oracle"
    assert receipt["active_output_comparisons"] == 0
    assert receipt["compiler_selected_output_qualification"] == "not_tested"
    if RTL.is_dir():
        assert audit(RTL) == receipt
