"""Keep the completed SmolVLA Spike session claim tied to its raw console."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/smolvla_full_session_spike_20261011"


def test_full_session_reaches_guest_done_with_every_output_and_state():
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    log = (ARCHIVE / "spike.log").read_bytes()
    assert hashlib.sha256(log).hexdigest() == summary["log_sha256"]
    assert summary["complete"] and summary["exit_code"] == 0
    assert summary["status"] == "complete_numeric_qualification_pending"
    assert summary["expected_events"] == summary["observed_events"] == 45
    lines = log.decode().splitlines()
    digests = [line for line in lines if line.startswith("SESSION_SHA ")]
    begins = [line for line in lines if line.startswith("SESSION_BEGIN ")]
    assert len(digests) == 33
    assert begins == ["SESSION_BEGIN 0 0", *(
        f"SESSION_BEGIN 1 {step}" for step in range(10)),
        "SESSION_BEGIN 2 0"]
    assert len(begins) + len(digests) == summary["observed_events"]
    assert "METRIC memref_rank_mismatch 0" in lines
    assert "DONE" in lines
