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


def measured_pair(root, capture, *, cycles, reset, prepare):
    from merlin.llvmlower.session_bundle import load

    from tools.run_linked_merlin_session import observe_preparation_output, observe_session_output

    link_path, run_path = _write_pair(root, cycles=cycles)
    link, run = (json.loads(p.read_text()) for p in (link_path, run_path))
    scope = "reset_plus_program_state_handoff_plus_existing_step_intervals; console_and_phase_bookkeeping_excluded"
    link.update(session_root=str(capture), performance_schema="spike_rdcycle_rdinstret_per_step.v1",
                preparation_performance_schema="spike_rdcycle_rdinstret_reset_prepare.v1", execution_timing_scope=scope)
    link_path.write_text(json.dumps(link))
    digest = hashlib.sha256(b"0000").hexdigest()
    console = (f"SESSION_PREP_PERF -1 {reset} {reset}\nSESSION_PREP_PERF 0 {prepare} {prepare}\n"
               f"SESSION_BEGIN 0 0\nSESSION_PERF 0 0 {cycles} {cycles}\nSESSION_SHA 0 0 0 4 {digest}\n"
               f"SESSION_TOTAL {cycles} {cycles}\nSESSION_EXECUTION_TOTAL {cycles + reset + prepare} {cycles + reset + prepare}\n"
               "METRIC memref_rank_mismatch 0\nDONE\n")
    (root / "spike.log").write_text(console)
    session = load(capture)
    observation = observe_session_output(console, session, exit_code=0, require_perf=True)
    run.update(observation | observe_preparation_output(console, session, observation))
    run.update(link_receipt_sha256=hashlib.sha256(link_path.read_bytes()).hexdigest(),
               log_sha256=hashlib.sha256((root / "spike.log").read_bytes()).hexdigest(),
               performance_schema=link["performance_schema"], execution_timing_scope=scope,
               preparation_performance_schema=link["preparation_performance_schema"])
    run_path.write_text(json.dumps(run))
    return link_path, run_path


@pytest.fixture
def capture(tmp_path):
    import yaml

    root = tmp_path / "capture"
    stage = root / "stage"
    stage.mkdir(parents=True)
    (stage / "model.mlir").write_text('''module {func.func @forward()->tensor<1xf32> {
      %c=arith.constant dense<0.0>:tensor<1xf32>
      func.return %c:tensor<1xf32>
    }}''')
    (stage / "inputs.npz").write_bytes(b"unused by ABI loader")
    (stage / "weights.safetensors.manifest.json").write_text("{}")
    (root / "session_contract.yaml").write_text(yaml.safe_dump({
        "version": 2, "stages": ["stage"], "programs": [{"name": "stage", "bundle": "stage", "steps": 1}],
        "stage_schedule": [{"name": "stage", "steps": 1, "execution": "compiled", "timed": True}],
        "bindings": [], "states": [], "quality": {"program": "stage"}}))
    return root


def test_faster_steps_can_be_slower_after_measured_preparation(tmp_path, capture):
    reference = measured_pair(tmp_path / "reference", capture, cycles=100, reset=7, prepare=4)
    candidate = measured_pair(tmp_path / "candidate", capture, cycles=90, reset=20, prepare=9)
    report = compare_sessions(*reference, *candidate)
    assert report["reference_cycles"] == 111 and report["candidate_cycles"] == 119
    assert report["candidate_over_reference"] == pytest.approx(119 / 111)
    assert report["programs"][0]["candidate_step_cycles"] == 90
    assert report["candidate_reset_cycles"] + sum(p["candidate_cycles"] for p in report["programs"]) == 119


@pytest.mark.parametrize("mutation", ["missing", "total", "console"])
def test_preparation_totals_must_reopen_against_actual_native_records(tmp_path, capture, mutation):
    reference = measured_pair(tmp_path / "reference", capture, cycles=100, reset=7, prepare=4)
    candidate = measured_pair(tmp_path / "candidate", capture, cycles=90, reset=7, prepare=4)
    run = json.loads(candidate[1].read_text())
    if mutation == "missing":
        run["preparation_performance"] = []
    elif mutation == "total":
        run["execution_total_performance"]["cycles"] -= 7
    else:
        console = candidate[1].parent / "spike.log"
        console.write_text(console.read_text().replace("SESSION_PERF 0 0 90 90", "SESSION_PERF 0 0 89 89"))
        run["log_sha256"] = hashlib.sha256(console.read_bytes()).hexdigest()
    candidate[1].write_text(json.dumps(run))
    with pytest.raises(ValueError):
        compare_sessions(*reference, *candidate)
