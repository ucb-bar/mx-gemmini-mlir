"""Keep the full VPU source-oracle receipt tied to its replay tool."""

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_vpu_ops_source import EXPECTED


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nicolas_all_vpu_source_ops_on_spike():
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    log = (EVIDENCE / "spike.log").read_text()
    assert receipt["status"] == "all_29_source_vpu_checks_matched_on_pinned_spike"
    assert receipt["scope"].endswith("source oracle only")
    assert receipt["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert receipt["qualifier_sha256"] == sha(ROOT / "tools/qualify_nicolas_vpu_ops_source.py")
    assert receipt["spike_log_sha256"] == sha(EVIDENCE / "spike.log")
    assert receipt["spike_exit_code"] == 0
    assert receipt["check_count"] == len(EXPECTED) == 29
    assert tuple(receipt["checks"]) == EXPECTED
    assert log.count("ok (0 mismatches)") == 29
    assert "vpu_ops PASSED" in log
