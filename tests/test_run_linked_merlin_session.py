"""The session runner binds every step and state digest to its capture ABI."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tools.run_linked_merlin_session import (
    extract_output_dumps, observe_session_output, session_progress,
)


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
    for marker in ("FATAL: model arena -- exhausted", "FAIL session state",
                   "FAIL session output ABI"):
        with pytest.raises(ValueError, match="complete valid session"):
            observe_session_output(_log() + marker + "\n", _session(), exit_code=0)


def test_per_step_cycles_and_instructions_cover_the_whole_session() -> None:
    log = _log().replace("SESSION_SHA 0 0 0", "SESSION_PERF 0 0 100 90\nSESSION_SHA 0 0 0")
    log = log.replace("SESSION_SHA 1 0 0", "SESSION_PERF 1 0 40 35\nSESSION_SHA 1 0 0")
    log = log.replace("SESSION_SHA 1 1 0", "SESSION_PERF 1 1 50 45\nSESSION_SHA 1 1 0")
    log = log.replace("METRIC memref", "SESSION_TOTAL 190 170\nMETRIC memref")
    report = observe_session_output(log, _session(), exit_code=0, require_perf=True)
    assert report["total_performance"] == {"cycles": 190, "instructions": 170}
    assert len(report["step_performance"]) == 3
    assert [row["cycles"] for row in report["program_performance"]] == [100, 90]
    assert session_progress(log, _session(), elapsed_wall_s=2.5)["completed_steps"] == 3
    with pytest.raises(ValueError, match="total performance differs"):
        observe_session_output(log.replace("SESSION_TOTAL 190", "SESSION_TOTAL 191"),
                               _session(), exit_code=0, require_perf=True)
    with pytest.raises(ValueError, match="complete valid session"):
        observe_session_output(_log(), _session(), exit_code=0, require_perf=True)


def test_selected_output_dump_is_bound_to_guest_digest(tmp_path) -> None:
    value = bytes(range(12))
    digest = hashlib.sha256(value).hexdigest()
    record = {"program": 1, "step": 0, "output": 0,
              "bytes": 12, "sha256": digest}
    selected = [{"program": 1, "output": 0}]
    session = SimpleNamespace(programs=(SimpleNamespace(steps=0),
                                        SimpleNamespace(steps=1)))
    line = f"SESSION_HEX 1 0 0 12 {value.hex()}\n"
    result = extract_output_dumps(line, session, selected, [record], tmp_path)
    assert result[0]["sha256"] == digest
    assert (tmp_path / result[0]["path"]).read_bytes() == value
    with pytest.raises(ValueError, match="guest digest"):
        extract_output_dumps(line.replace(value.hex(), bytes(reversed(value)).hex()),
                             session, selected, [record], tmp_path)
    with pytest.raises(ValueError, match="lacks a selected output dump"):
        extract_output_dumps("", session, selected, [record], tmp_path)
