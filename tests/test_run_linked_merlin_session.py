"""The session runner binds every step and state digest to its capture ABI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.run_linked_merlin_session import observe_session_output


def _session():
    return SimpleNamespace(programs=(
        SimpleNamespace(steps=1, outputs=(([2], "i1"), ([2, 3], "bf16"))),
        SimpleNamespace(steps=2, outputs=(([3], "f32"),)),
    ))


def _log() -> str:
    digest = "a" * 64
    return "\n".join((
        "SESSION_BEGIN 0 0", f"SESSION_SHA 0 0 0 2 {digest}",
        f"SESSION_SHA 0 0 1 12 {digest}",
        "SESSION_BEGIN 1 0", f"SESSION_SHA 1 0 0 12 {digest}",
        "SESSION_BEGIN 1 1", f"SESSION_SHA 1 1 0 12 {digest}",
        "METRIC memref_rank_mismatch 0", "DONE", "",
    ))


def test_complete_session_checks_all_programs_and_state_outputs() -> None:
    report = observe_session_output(_log(), _session(), exit_code=0)
    assert report["complete"]
    assert report["observed_events"] == report["expected_events"] == 7
    assert len(report["output_digests"]) == 4


def test_completed_run_refuses_wrong_state_bytes_and_missing_records() -> None:
    with pytest.raises(ValueError, match="result digest differs"):
        observe_session_output(_log().replace("0 0 1 12", "0 0 1 8"),
                               _session(), exit_code=0)
    with pytest.raises(ValueError, match="complete valid session"):
        observe_session_output(_log().split("SESSION_BEGIN 1 1")[0]
                               + "METRIC memref_rank_mismatch 0\nDONE\n",
                               _session(), exit_code=0)


def test_timeout_preserves_only_a_valid_prefix() -> None:
    log = _log().split("SESSION_BEGIN 1 0")[0] + "SESSION_SHA 1"
    report = observe_session_output(log, _session(), exit_code=124)
    assert not report["complete"]
    assert report["observed_events"] == 3
    assert len(report["output_digests"]) == 2


def test_guest_arena_failure_is_not_hidden_by_spike_exit_zero() -> None:
    guest_log = "SESSION_BEGIN 0 0\nFATAL model arena exhausted\n*** FAILED ***\n"
    with pytest.raises(ValueError, match="complete valid session"):
        observe_session_output(guest_log, _session(), exit_code=0)
    with pytest.raises(ValueError, match="complete valid session"):
        observe_session_output(_log() + "FATAL model arena exhausted\n",
                               _session(), exit_code=0)
