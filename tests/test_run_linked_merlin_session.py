"""The session runner binds every step and state digest to its capture ABI."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.run_linked_merlin_session import (
    extract_output_dumps,
    observe_preparation_output,
    observe_session_output,
    session_progress,
    validate_admission_record,
    validate_linked_layout,
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


def test_runner_rejects_tinyllama_arena_overlapping_linked_elf() -> None:
    layout = {"arena_base": 0xC0000000, "arena_bytes": 2 << 30,
              "weights_base": 0x200000000, "mem_bytes": 11 << 30}
    check = dict(mem_bytes=11 << 30, weight_bytes=4_400_193_536,
                 measured_static_end=0xC49108F0)
    with pytest.raises(ValueError, match="overlaps static data"):
        validate_linked_layout(layout, **check)
    layout["arena_base"] = 0xD0000000
    layout["static_end"] = check["measured_static_end"]
    validate_linked_layout(layout, **check)


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


def _preparation_log():
    log = _log().replace("SESSION_SHA 0 0 0", "SESSION_PERF 0 0 100 90\nSESSION_SHA 0 0 0")
    log = log.replace("SESSION_SHA 1 0 0", "SESSION_PERF 1 0 40 35\nSESSION_SHA 1 0 0")
    log = log.replace("SESSION_SHA 1 1 0", "SESSION_PERF 1 1 50 45\nSESSION_SHA 1 1 0")
    log = "SESSION_PREP_PERF -1 10 1\nSESSION_PREP_PERF 0 20 2\n" + log
    log = log.replace("SESSION_BEGIN 1 0", "SESSION_PREP_PERF 1 30 3\nSESSION_BEGIN 1 0")
    return log.replace("METRIC memref", "SESSION_TOTAL 190 170\nSESSION_EXECUTION_TOTAL 250 176\nMETRIC memref")


def test_preparation_accounting_adds_real_work_without_replacing_step_intervals():
    log = _preparation_log()
    original = observe_session_output(log, _session(), exit_code=0, require_perf=True)
    measured = observe_preparation_output(log, _session(), original)
    assert original["total_performance"] == {"cycles": 190, "instructions": 170}
    assert measured["preparation_total_performance"] == {"cycles": 60, "instructions": 6}
    assert measured["execution_total_performance"] == {"cycles": 250, "instructions": 176}


@pytest.mark.parametrize("change", [
    lambda log: log.replace("SESSION_PREP_PERF 1 30 3\n", ""),
    lambda log: log.replace("SESSION_EXECUTION_TOTAL 250", "SESSION_EXECUTION_TOTAL 190"),
    lambda log: "SESSION_PREP_PERF -1 10 1\n" + log,
    lambda log: log.replace("SESSION_PREP_PERF 0 20 2", "SESSION_PREP_PERF 0 20 0"),
    lambda log: log.replace("SESSION_EXECUTION_TOTAL 250 176\n", ""),
    lambda log: log.replace("SESSION_PREP_PERF 1 30 3\n", "").replace(
        "SESSION_BEGIN 0 0", "SESSION_PREP_PERF 1 30 3\nSESSION_BEGIN 0 0"),
])
def test_preparation_metrics_refuse_missing_or_misplaced_costs(change):
    log = change(_preparation_log())
    original = observe_session_output(log, _session(), exit_code=0, require_perf=True)
    with pytest.raises(ValueError):
        observe_preparation_output(log, _session(), original)


def test_required_final_admission_reopens_the_actual_policy_report(tmp_path):
    report = tmp_path / "admission.json"
    stored = {"status": "accepted", "elf_sha256": "a" * 64}
    report.write_text(json.dumps(stored))
    record = {"elf_sha256": "a" * 64, "linked_elf_admission_required": True,
              "linked_elf_admission": {**stored, "report_path": str(report),
                                       "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}}
    validate_admission_record(record)
    report.write_text(json.dumps({**stored, "status": "refused"}))
    with pytest.raises(ValueError, match="report changed"):
        validate_admission_record(record)
    with pytest.raises(ValueError, match="required final ELF admission"):
        validate_admission_record({"linked_elf_admission_required": True})


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


def test_large_logits_dump_requires_matching_link_limit(tmp_path) -> None:
    value = bytes(range(256)) * 500
    record = {"program": 0, "step": 0, "output": 0,
              "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    session = SimpleNamespace(programs=(SimpleNamespace(steps=1),))
    selected = [{"program": 0, "output": 0}]
    line = f"SESSION_HEX 0 0 0 {len(value)} {value.hex()}\n"
    with pytest.raises(ValueError, match="capture ABI"):
        extract_output_dumps(line, session, selected, [record], tmp_path)
    result = extract_output_dumps(line, session, selected, [record], tmp_path,
                                  byte_limit=len(value))
    assert result[0]["bytes"] == len(value)
    assert (tmp_path / result[0]["path"]).read_bytes() == value
