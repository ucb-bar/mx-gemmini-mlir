"""Synthetic protocol fixtures test evidence validation, never model execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from merlin.llvmlower.session_bundle import MultiProgramSession, Program

from tools.evaluate_matched_tinyllama_session import (
    trajectory_metrics,
    verify_completed_run,
    verify_vpu_policy_bindings,
)
from tools.run_linked_merlin_session import (
    observe_preparation_output,
    observe_session_output,
)


def write_json(path, data):
    path.write_text(json.dumps(data))


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    outputs = (([1, 32000], "f32"), ([1], "f32"), ([1], "i64"))
    programs = tuple(Program(name, tmp_path, steps, (), outputs)
                     for name, steps in (("prefill", 1), ("decode", 32)))
    session = MultiProgramSession(tmp_path, {}, programs, (), "decode", 0)
    actual = np.linspace(-2, 2, 32000, dtype=np.float32).reshape(1, -1)
    raw = actual.tobytes()
    command = []
    for name, contents in (("spike", b"synthetic simulator identity"),
                           ("extension.so", b"synthetic extension identity")):
        path = tmp_path / name
        path.write_bytes(contents)
        command.append(str(path))
    image = tmp_path / "session.elf"
    image.write_bytes(b"synthetic unit fixture, not an executed ELF")
    command = [command[0], "--extlib=" + command[1], str(image)]
    report = tmp_path / "admission.json"
    admitted = {"status": "accepted", "elf_sha256": digest(image.read_bytes())}
    write_json(report, admitted)
    taps = [{"program": p, "name": name, "output": 0, "bytes": 128000}
            for p, name in enumerate(("prefill", "decode"))]
    scope = "reset_plus_program_state_handoff_plus_existing_step_intervals; console_and_phase_bookkeeping_excluded"
    link = {"elf": str(image), "elf_bytes": image.stat().st_size,
            "elf_sha256": digest(image.read_bytes()), "performance_schema": "spike_rdcycle_rdinstret_per_step.v1",
            "preparation_performance_schema": "spike_rdcycle_rdinstret_reset_prepare.v1",
            "execution_timing_scope": scope, "dump_outputs": taps, "dump_limit_bytes": 128000,
            "linked_elf_admission_required": True,
            "linked_elf_admission": {**admitted, "report_path": str(report), "report_sha256": digest(report.read_bytes())}}
    link_path = tmp_path / "link.json"
    write_json(link_path, link)
    lines, dumps, total, instructions = ["gemmini: GEMMINI_MODE=both", "SESSION_PREP_PERF -1 10 9"], [], 0, 0
    dump_dir = tmp_path / "output_dumps"
    dump_dir.mkdir()
    for p, program in enumerate(programs):
        lines.append(f"SESSION_PREP_PERF {p} 5 4")
        for step in range(program.steps):
            cycles = 100 + step + p
            total += cycles
            instructions += cycles - 1
            lines.extend((f"SESSION_BEGIN {p} {step}", f"SESSION_PERF {p} {step} {cycles} {cycles - 1}"))
            for output, value in enumerate((raw, b"1234", b"12345678")):
                lines.append(f"SESSION_SHA {p} {step} {output} {len(value)} {digest(value)}")
            lines.append(f"SESSION_HEX {p} {step} 0 {len(raw)} {raw.hex()}")
            filename = f"output_dumps/p{p}_s{step}_o0.bin"
            (tmp_path / filename).write_bytes(raw)
            dumps.append({"program": p, "step": step, "output": 0, "path": filename,
                          "bytes": len(raw), "sha256": digest(raw)})
    lines.extend((f"SESSION_TOTAL {total} {instructions}", f"SESSION_EXECUTION_TOTAL {total + 20} {instructions + 17}",
                  "METRIC memref_rank_mismatch 0", "DONE", ""))
    console = "\n".join(lines)
    (tmp_path / "spike.log").write_text(console)
    perf = tmp_path / "gemmini_perf.txt"
    perf.write_text("gemmini perf [e4m3_vpu]: 9 commands, 1 loops, 3 fences; last activity 100; host stalled 0 cycles; 10 events\n"
                    "  mesh: 1 tiles, 16 busy cycles; VPU: 6 commands\n"
                    "  load: 128 bytes in 4 Gets (mean latency 1); store: 1 reads, 1 Puts; scales: 8 bytes\n")
    write_json(tmp_path / "live_environment.json", {"command": command, "environment": {
        "GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
        "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(perf)}})
    observation = observe_session_output(console, session, exit_code=0, require_perf=True)
    run = {"schema": "mx_gemmini.merlin_session_spike_run.v1", "status": "complete_numeric_qualification_pending",
           "complete": True, "exit_code": 0, "link_receipt_sha256": digest(link_path.read_bytes()),
           "timing_mode": "both", "timing_config": "e4m3_vpu", "timing_overrides": "mem.host_tracking=0",
           "timing_model_active": True, "command": command, "spike_sha256": digest(Path(command[0]).read_bytes()),
           "extension_sha256": digest((tmp_path / "extension.so").read_bytes()),
           "performance_schema": link["performance_schema"], "execution_timing_scope": scope,
           "preparation_performance_schema": link["preparation_performance_schema"],
           "log_sha256": digest((tmp_path / "spike.log").read_bytes()),
           "timing_report_sha256": digest(perf.read_bytes()), "output_dumps": dumps,
           **observation, **observe_preparation_output(console, session, observation)}
    run_path = tmp_path / "run.json"
    write_json(run_path, run)
    return session, run_path, link_path, run, np.repeat(actual, 33, axis=0)


def test_reopen_every_row_and_include_reset_and_handoff(evidence):
    session, run_path, link_path, run, expected = evidence
    actual_run, actual, perf = verify_completed_run(run_path, link_path, session)
    assert np.array_equal(actual, expected)
    assert actual_run["execution_total_performance"]["cycles"] == run["total_performance"]["cycles"] + 20
    assert perf["vpu_commands"] == 6
    quality = trajectory_metrics(actual, expected)
    assert quality["mean_kl_nats"] == 0 and quality["top_token_matches"] == 33
    assert [r["step"] for r in quality["rows"][1:]] == list(range(32))


@pytest.mark.parametrize("mutation", ["incomplete", "step_only", "cycle_total", "missing_output", "log",
                                    "image", "extension", "dump", "symlink", "missing_dump", "admission", "timing", "console_tap"])
def test_refuse_changed_partial_and_incomparable_execution_evidence(evidence, mutation):
    session, run_path, link_path, run, _ = evidence
    root = run_path.parent
    if mutation == "incomplete":
        run["complete"] = False
    elif mutation == "step_only":
        run["preparation_performance_schema"] = None
    elif mutation == "cycle_total":
        run["execution_total_performance"]["cycles"] -= 10
    elif mutation == "missing_output":
        run["output_digests"].pop()
    elif mutation == "missing_dump":
        run["output_dumps"].pop()
    elif mutation == "dump":
        (root / run["output_dumps"][0]["path"]).write_bytes(b"changed")
    elif mutation == "symlink":
        original = root / run["output_dumps"][0]["path"]
        alias = root / "alias.bin"
        alias.symlink_to(original)
        run["output_dumps"][0]["path"] = alias.name
    elif mutation == "console_tap":
        console_path = root / "spike.log"
        console = console_path.read_text()
        line = next(line for line in console.splitlines() if line.startswith("SESSION_HEX 0 0 0 "))
        parts = line.split()
        parts[-1] = "ffffffff" + parts[-1][8:]
        console_path.write_text(console.replace(line, " ".join(parts), 1))
        run["log_sha256"] = digest(console_path.read_bytes())
    else:
        name = {"log": "spike.log", "image": "session.elf", "extension": "extension.so",
                "admission": "admission.json", "timing": "gemmini_perf.txt"}[mutation]
        (root / name).write_bytes(b"changed")
    write_json(run_path, run)
    with pytest.raises(ValueError):
        verify_completed_run(run_path, link_path, session)


def test_kl_and_token_agreement_are_diagnostic_without_a_source_gate():
    reference = np.zeros((33, 32000), np.float32)
    reference[:, 7] = 2
    actual = reference.copy()
    actual[4, 11] = 3
    result = trajectory_metrics(actual, reference)
    assert result["top_token_matches"] == 32 and result["worst_kl_nats"] > 0
    assert "passed" not in result
    actual[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        trajectory_metrics(actual, reference)


@pytest.mark.parametrize("mutation", [None, "extra", "missing", "count", "alias"])
def test_vpu_coverage_follows_explicit_policy_and_verified_scalar_add_alias(mutation):
    enabled = {name: frozenset() for name in ("softmax", "add", "elementwise", "unary", "reduce")}
    enabled["add"] = frozenset({"bias"})
    enabled["unary"] = frozenset({"activation#sigmoid0"})
    routed = {name: sorted(values) for name, values in enabled.items()}
    routed["add"], routed["elementwise"] = [], ["bias#add0"]
    manifest = {f"vpu_{name}": {"source_regions": regions.copy()} for name, regions in routed.items()}
    manifest.update(resolved_vpu_regions=routed, vpu_source_aliases={"bias": "bias#add0"})
    receipt = {f"vpu_{name}_lowered_regions": len(values) for name, values in routed.items()}
    if mutation == "extra":
        manifest["vpu_unary"]["source_regions"].append("unselected#sigmoid0")
    elif mutation == "missing":
        manifest["vpu_unary"]["source_regions"] = []
    elif mutation == "count":
        receipt["vpu_unary_lowered_regions"] = 0
    elif mutation == "alias":
        manifest["vpu_source_aliases"] = {}
    if mutation is None:
        verify_vpu_policy_bindings(manifest, {"enabled_vpu_regions": enabled}, receipt)
    else:
        with pytest.raises(ValueError):
            verify_vpu_policy_bindings(manifest, {"enabled_vpu_regions": enabled}, receipt)
