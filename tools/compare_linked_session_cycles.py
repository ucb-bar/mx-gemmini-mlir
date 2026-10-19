"""Compare two completed, equal-input linked MX sessions on the same Spike model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.spike_timing import reject_deadlocked_timing


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(link_path: Path, run_path: Path) -> tuple[dict, dict]:
    link = json.loads(link_path.read_text())
    run = json.loads(run_path.read_text())
    if (link.get("schema") != "mx_gemmini.linked_merlin_session.v1" or
            run.get("schema") != "mx_gemmini.merlin_session_spike_run.v1" or
            run.get("link_receipt_sha256") != _sha(link_path) or
            run.get("status") != "complete_numeric_qualification_pending" or
            run.get("complete") is not True or run.get("exit_code") != 0 or
            not run.get("timing_model_active") or
            not isinstance(run.get("total_performance"), dict)):
        raise ValueError("comparison needs complete receipt-bound timing runs")
    if _sha(run_path.parent / "spike.log") != run.get("log_sha256"):
        raise ValueError("Spike log differs from its run receipt")
    reject_deadlocked_timing((run_path.parent / "spike.log").read_text())
    command = run.get("command", [])
    if (not command or Path(command[-1]).resolve() != Path(link.get("elf", "")).resolve()):
        raise ValueError("Spike command does not run the linked ELF")
    steps = run.get("step_performance", [])
    total = run["total_performance"]
    if (len(steps) != sum(row["steps"] for row in link["programs"]) or
            len(run.get("output_digests", [])) != sum(
                row["steps"] * row["outputs"] for row in link["programs"]) or
            any(not isinstance(total.get(key), int) or total[key] <= 0
                for key in ("cycles", "instructions")) or
            any(total.get(key) != sum(row[key] for row in steps)
                for key in ("cycles", "instructions"))):
        raise ValueError("session timing does not reconcile with step records")
    preparation_schema = link.get("preparation_performance_schema")
    if preparation_schema is not None:
        from merlin.llvmlower.session_bundle import load

        from tools.run_linked_merlin_session import (
            observe_preparation_output,
            observe_session_output,
        )

        if (preparation_schema != "spike_rdcycle_rdinstret_reset_prepare.v1"
                or run.get("preparation_performance_schema") != preparation_schema
                or link.get("execution_timing_scope") != "reset_plus_program_state_handoff_plus_existing_step_intervals; console_and_phase_bookkeeping_excluded"
                or run.get("execution_timing_scope") != link.get("execution_timing_scope")):
            raise ValueError("session preparation timing selection differs")
        session = load(Path(link["session_root"]))
        console = (run_path.parent / "spike.log").read_text()
        observed = observe_session_output(console, session, exit_code=0, require_perf=True)
        preparation = observe_preparation_output(console, session, observed)
        if any(run.get(key) != value for key, value in (observed | preparation).items()):
            raise ValueError("session measured execution differs from the actual console")
        run["comparison_performance"] = preparation["execution_total_performance"]
    else:
        if (run.get("preparation_performance_schema") is not None
                or run.get("execution_total_performance") is not None
                or run.get("preparation_performance") or run.get("preparation_total_performance") is not None):
            raise ValueError("unselected preparation intervals cannot change session timing")
        run["comparison_performance"] = total
    return link, run


def compare_sessions(reference_link: Path, reference_run: Path,
                     candidate_link: Path, candidate_run: Path) -> dict:
    base_link, base = _load(reference_link, reference_run)
    other_link, other = _load(candidate_link, candidate_run)
    identity = ("session_root", "programs", "bindings", "profile_sha256",
                "shared_weight_sha256", "shared_weight_bytes", "mem_bytes",
                "performance_schema", "preparation_performance_schema", "execution_timing_scope")
    if any(base_link.get(key) != other_link.get(key) for key in identity):
        raise ValueError("linked sessions differ in captured work, weights, or profile")
    timing = ("spike_sha256", "extension_sha256", "timing_mode",
              "timing_config", "timing_overrides", "performance_schema")
    if any(base.get(key) != other.get(key) for key in timing):
        raise ValueError("runs use different Spike or timing settings")
    if base["command"][:-1] != other["command"][:-1]:
        raise ValueError("runs use different Spike launch arguments")
    outputs = ("program", "step", "output", "bytes", "sha256")
    base_outputs = [tuple(row[key] for key in outputs) for row in base["output_digests"]]
    other_outputs = [tuple(row[key] for key in outputs) for row in other["output_digests"]]
    if base_outputs != other_outputs:
        raise ValueError("session output digests differ; cycles are not equal-work evidence")
    base_steps = base["step_performance"]
    other_steps = other["step_performance"]
    if [(row["program"], row["step"]) for row in base_steps] != [
            (row["program"], row["step"]) for row in other_steps]:
        raise ValueError("session step schedules differ")
    base_objects = base_link.get("stage_object_sha256", {})
    other_objects = other_link.get("stage_object_sha256", {})
    if set(base_objects) != set(other_objects) or any(
            set(base_objects[stage]) != set(other_objects[stage])
            for stage in base_objects):
        raise ValueError("linked sessions have different stage object sets")
    changed_objects = [
        f"{stage}/{name}" for stage in sorted(base_objects)
        for name in sorted(base_objects[stage])
        if base_objects[stage][name] != other_objects[stage][name]
    ]
    programs = []
    for index, program in enumerate(base_link["programs"]):
        base_step_cycles = sum(row["cycles"] for row in base_steps if row["program"] == index)
        other_step_cycles = sum(row["cycles"] for row in other_steps if row["program"] == index)
        base_preparation = sum(row["cycles"] for row in base.get("preparation_performance", []) if row["program"] == index)
        other_preparation = sum(row["cycles"] for row in other.get("preparation_performance", []) if row["program"] == index)
        base_cycles, other_cycles = base_step_cycles + base_preparation, other_step_cycles + other_preparation
        programs.append({"name": program["name"], "steps": program["steps"],
                         "reference_cycles": base_cycles,
                         "candidate_cycles": other_cycles,
                         "reference_step_cycles": base_step_cycles,
                         "candidate_step_cycles": other_step_cycles,
                         "reference_preparation_cycles": base_preparation,
                         "candidate_preparation_cycles": other_preparation,
                         "candidate_over_reference": other_cycles / base_cycles})
    return {
        "schema": "mx_gemmini.equal_input_session_cycles.v1",
        "reference_link_sha256": _sha(reference_link),
        "candidate_link_sha256": _sha(candidate_link),
        "reference_run_sha256": _sha(reference_run),
        "candidate_run_sha256": _sha(candidate_run),
        "output_digest_parity": True,
        "changed_stage_objects": changed_objects,
        **({"session_host_math": {
            "reference": (base_link.get("host_math") or {}).get("contract"),
            "candidate": (other_link.get("host_math") or {}).get("contract"),
        }} if base_link.get("host_math") or other_link.get("host_math") else {}),
        "spike_sha256": base["spike_sha256"],
        "extension_sha256": base["extension_sha256"],
        "profile_sha256": base_link["profile_sha256"],
        "timing_config": base["timing_config"],
        "timing_overrides": base["timing_overrides"],
        "execution_timing_scope": base_link.get("execution_timing_scope", "legacy_step_intervals_only"),
        "reference_reset_cycles": sum(row["cycles"] for row in base.get("preparation_performance", []) if row["program"] == -1),
        "candidate_reset_cycles": sum(row["cycles"] for row in other.get("preparation_performance", []) if row["program"] == -1),
        "reference_cycles": base["comparison_performance"]["cycles"],
        "candidate_cycles": other["comparison_performance"]["cycles"],
        "candidate_over_reference": (other["comparison_performance"]["cycles"] /
                                     base["comparison_performance"]["cycles"]),
        "programs": programs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_link", "reference_run", "candidate_link", "candidate_run", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    report = compare_sessions(args.reference_link, args.reference_run,
                              args.candidate_link, args.candidate_run)
    if args.out.exists():
        parser.error("--out must be a new file")
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "reference_cycles", "candidate_cycles", "candidate_over_reference")}))


if __name__ == "__main__":
    main()
