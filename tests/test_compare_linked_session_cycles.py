"""Whole-session cycle comparisons require the same captured execution."""

from __future__ import annotations

import hashlib
import json

import pytest

from tools.compare_linked_session_cycles import compare_sessions


def _write_pair(root, *, cycles: int):
    root.mkdir()
    link_path = root / "link.json"
    link_path.write_text(json.dumps({
        "schema": "mx_gemmini.linked_merlin_session.v1",
        "elf": str(root / "session.elf"),
        "session_root": "/capture", "profile_sha256": "profile",
        "shared_weight_sha256": "weights", "shared_weight_bytes": 8,
        "mem_bytes": 1024, "performance_schema": "per_step",
        "bindings": [],
        "stage_object_sha256": {"stage": {"model.o": "host", "device_catalog/mx_catalog.o": root.name}},
        "programs": [{"name": "stage", "steps": 1, "outputs": 1,
                      "capture_model_sha256": "source", "selected_operations": 1}],
    }))
    log = root / "spike.log"
    log.write_text("DONE\n")
    run_path = root / "run.json"
    run_path.write_text(json.dumps({
        "schema": "mx_gemmini.merlin_session_spike_run.v1",
        "command": ["spike", "--isa=rv64gc", str(root / "session.elf")],
        "link_receipt_sha256": hashlib.sha256(link_path.read_bytes()).hexdigest(),
        "status": "complete_numeric_qualification_pending", "complete": True,
        "exit_code": 0, "timing_model_active": True,
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "spike_sha256": "spike", "extension_sha256": "extension",
        "timing_mode": "both", "timing_config": "e4m3_vpu",
        "timing_overrides": "mem.host_tracking=0", "performance_schema": "per_step",
        "step_performance": [{"program": 0, "step": 0, "cycles": cycles,
                              "instructions": cycles}],
        "total_performance": {"cycles": cycles, "instructions": cycles},
        "output_digests": [{"program": 0, "step": 0, "output": 0,
                            "bytes": 4, "sha256": "value"}],
    }))
    return link_path, run_path


def test_equal_input_sessions_compare_stage_and_total_cycles(tmp_path) -> None:
    reference = _write_pair(tmp_path / "reference", cycles=100)
    candidate = _write_pair(tmp_path / "candidate", cycles=90)
    report = compare_sessions(*reference, *candidate)
    assert report["output_digest_parity"]
    assert report["candidate_over_reference"] == 0.9
    assert report["programs"][0]["candidate_cycles"] == 90
    assert report["changed_stage_objects"] == ["stage/device_catalog/mx_catalog.o"]

    run = json.loads(candidate[1].read_text())
    run["output_digests"][0]["sha256"] = "different"
    candidate[1].write_text(json.dumps(run))
    with pytest.raises(ValueError, match="output digests differ"):
        compare_sessions(*reference, *candidate)


def test_changed_capture_or_timing_model_is_rejected(tmp_path) -> None:
    reference = _write_pair(tmp_path / "reference", cycles=100)
    candidate = _write_pair(tmp_path / "candidate", cycles=90)
    link = json.loads(candidate[0].read_text())
    link["profile_sha256"] = "other"
    candidate[0].write_text(json.dumps(link))
    with pytest.raises(ValueError, match="receipt-bound"):
        compare_sessions(*reference, *candidate)
    run = json.loads(candidate[1].read_text())
    run["link_receipt_sha256"] = hashlib.sha256(candidate[0].read_bytes()).hexdigest()
    candidate[1].write_text(json.dumps(run))
    with pytest.raises(ValueError, match="captured work"):
        compare_sessions(*reference, *candidate)
    link["profile_sha256"] = "profile"
    candidate[0].write_text(json.dumps(link))
    run["link_receipt_sha256"] = hashlib.sha256(candidate[0].read_bytes()).hexdigest()
    run["timing_config"] = "mx_rocket"
    candidate[1].write_text(json.dumps(run))
    with pytest.raises(ValueError, match="timing settings"):
        compare_sessions(*reference, *candidate)
