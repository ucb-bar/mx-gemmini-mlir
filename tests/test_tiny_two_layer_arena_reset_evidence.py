"""Pinned logs establish the recurrent arena reset's first failure crossing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/tiny_two_layer_arena_reset_crossing_20261011"


def test_reset_crosses_prior_guest_arena_failure_with_same_valid_prefix() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    old = (EVIDENCE / "without_reset.log").read_text()
    new = (EVIDENCE / "with_reset_valid_prefix.log").read_text()
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    assert summary["schema"] == "mx_gemmini.tiny_two_layer_arena_reset_crossing.v1"
    assert summary["without_reset"]["log_sha256"] == digest(old)
    assert summary["with_reset"]["valid_prefix_log_sha256"] == digest(new)
    assert summary["without_reset"]["process_exit_code"] == 0
    assert "SESSION_BEGIN 1 2\n\nFATAL: model arena -- exhausted" in old
    assert "*** FAILED ***" in old
    old_sha = [line for line in old.splitlines() if line.startswith("SESSION_SHA ")]
    new_sha = [line for line in new.splitlines() if line.startswith("SESSION_SHA ")]
    assert len(old_sha) == 9 and len(new_sha) == 12
    assert old_sha == new_sha[:9]
    assert [line.split()[1:4] for line in new_sha[9:]] == [
        ["1", "2", str(output)] for output in range(3)]
    assert [line.split()[-1] for line in new_sha[9:]] == summary["step_2_output_digests"]
    assert "FATAL" not in new and "DONE" not in new
