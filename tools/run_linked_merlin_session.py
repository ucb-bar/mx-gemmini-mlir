"""Run a linked MX/Merlin session on pinned Spike and verify its result ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .link_merlin_session import (
    DEFAULT_DUMP_LIMIT_BYTES, MAX_DUMP_LIMIT_BYTES, _symbol_address,
)


_BYTES = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2, "f16": 2, "i32": 4}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_linked_layout(layout: dict, *, mem_bytes: int,
                           weight_bytes: int | None,
                           measured_static_end: int) -> None:
    """Reject an arena that can overwrite the ELF, weights, or Spike memory."""
    arena_start = int(layout["arena_base"])
    arena_end = arena_start + int(layout["arena_bytes"])
    weight_start = int(layout["weights_base"])
    weight_end = weight_start + weight_bytes if weight_bytes is not None else weight_start
    recorded_static_end = layout.get("static_end")
    if recorded_static_end is not None and int(recorded_static_end) != measured_static_end:
        raise ValueError("linked session static extent differs from its receipt")
    if (arena_start < 0x80000000 or arena_end > 0x80000000 + mem_bytes
            or arena_start < measured_static_end
            or max(arena_start, weight_start) < min(arena_end, weight_end)
            or int(layout["mem_bytes"]) > mem_bytes):
        raise ValueError("linked session arena overlaps static data or weights, "
                         "or exceeds Spike memory")


def _expected(session) -> list[tuple[str, int, int, int | None, int | None]]:
    events = []
    for program_index, program in enumerate(session.programs):
        for step in range(program.steps):
            events.append(("begin", program_index, step, None, None))
            for output_index, (shape, dtype) in enumerate(program.outputs):
                count = 1
                for dim in shape:
                    count *= dim
                if dtype not in _BYTES or count <= 0:
                    raise ValueError(f"unsupported session result shape/dtype: {shape}, {dtype}")
                events.append(("sha", program_index, step, output_index,
                               count * _BYTES[dtype]))
    return events


def observe_session_output(
    log: str, session, *, exit_code: int, require_perf: bool = False,
) -> dict:
    """Accept only a correct prefix of the captured session's output/state ABI."""
    expected = _expected(session)
    seen = 0
    records = []
    performance = []
    active_step = None
    step_has_output = False
    total = None
    for raw in log.splitlines(keepends=True):
        if exit_code != 0 and not raw.endswith("\n"):
            break  # A timed-out Spike process may stop between HTIF characters.
        line = raw.rstrip("\n")
        parts = line.split()
        if not parts or parts[0] not in {
            "SESSION_BEGIN", "SESSION_SHA", "SESSION_PERF", "SESSION_TOTAL"
        }:
            continue
        if parts[0] == "SESSION_PERF":
            if (len(parts) != 5 or active_step is None or step_has_output
                    or any(row["program"] == active_step[0] and
                           row["step"] == active_step[1] for row in performance)):
                raise ValueError("session performance record order differs from capture ABI")
            values = [int(value) for value in parts[1:]]
            if values[:2] != list(active_step) or min(values[2:]) <= 0:
                raise ValueError("session performance record differs from capture ABI")
            performance.append(dict(zip(
                ("program", "step", "cycles", "instructions"), values, strict=True)))
            continue
        if parts[0] == "SESSION_TOTAL":
            if len(parts) != 3 or total is not None or seen != len(expected):
                raise ValueError("session total performance record is out of order")
            total = {"cycles": int(parts[1]), "instructions": int(parts[2])}
            if (len(performance) != sum(program.steps for program in session.programs)
                    or any(total[key] != sum(row[key] for row in performance)
                           for key in total)):
                raise ValueError("session total performance differs from step records")
            continue
        if seen >= len(expected):
            raise ValueError("session emitted more result records than its capture ABI")
        kind, program, step, output, bytes_expected = expected[seen]
        if parts[0] == "SESSION_BEGIN":
            if len(parts) != 3 or kind != "begin" or [int(x) for x in parts[1:]] != [program, step]:
                raise ValueError("session program/step order differs from capture ABI")
            active_step = (program, step)
            step_has_output = False
        else:
            if (len(parts) != 6 or kind != "sha" or
                    [int(x) for x in parts[1:5]] != [program, step, output, bytes_expected]
                    or not _SHA.fullmatch(parts[5])):
                raise ValueError("session result digest differs from capture ABI")
            records.append({"program": program, "step": step, "output": output,
                            "bytes": bytes_expected, "sha256": parts[5]})
            step_has_output = True
        seen += 1
    done = "DONE" in log.splitlines()
    metrics = [line for line in log.splitlines()
               if line.startswith("METRIC memref_rank_mismatch ")]
    guest_failure = any(line.startswith(("FATAL:", "FATAL ", "FAIL session"))
                        or "*** FAILED" in line
                        for line in log.splitlines())
    complete = (not guest_failure and exit_code == 0 and done
                and seen == len(expected)
                and metrics == ["METRIC memref_rank_mismatch 0"]
                and (not require_perf or total is not None))
    if exit_code == 0 and not complete:
        raise ValueError("Spike exited successfully without a complete valid session")
    if done and not complete:
        raise ValueError("session printed DONE with incomplete or invalid result ABI")
    programs = [{
        "program": index,
        "name": getattr(program, "name", str(index)),
        "measured_steps": sum(row["program"] == index for row in performance),
        "cycles": sum(row["cycles"] for row in performance if row["program"] == index),
        "instructions": sum(row["instructions"] for row in performance
                            if row["program"] == index),
    } for index, program in enumerate(session.programs)]
    return {"complete": complete, "observed_events": seen,
            "expected_events": len(expected), "output_digests": records,
            "step_performance": performance, "program_performance": programs,
            "total_performance": total}


def session_progress(log: str, session, *, elapsed_wall_s: float) -> dict:
    """Describe the last completed and active steps in a running Spike log."""
    completed = 0
    active = None
    performance = []
    for line in log.splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        parts = line.split()
        if len(parts) == 3 and parts[0] == "SESSION_BEGIN":
            active = {"program": int(parts[1]), "step": int(parts[2])}
        elif len(parts) == 5 and parts[0] == "SESSION_PERF":
            performance.append(dict(zip(
                ("program", "step", "cycles", "instructions"),
                (int(value) for value in parts[1:]), strict=True)))
        elif (len(parts) == 6 and parts[0] == "SESSION_SHA"
              and _SHA.fullmatch(parts[5]) and active is not None):
            program = session.programs[active["program"]]
            if int(parts[3]) == len(program.outputs) - 1:
                completed += 1
                active = None
    return {
        "elapsed_wall_s": elapsed_wall_s,
        "completed_steps": completed,
        "total_steps": sum(program.steps for program in session.programs),
        "active_step": active,
        "step_performance": performance,
        "modeled_cycles_so_far": sum(row["cycles"] for row in performance),
    }


def extract_output_dumps(log: str, session, selected: list[dict],
                         digests: list[dict], out_dir: Path,
                         byte_limit: int = DEFAULT_DUMP_LIMIT_BYTES) -> list[dict]:
    """Save bounded guest values only when their bytes match the session digest."""
    if not DEFAULT_DUMP_LIMIT_BYTES <= byte_limit <= MAX_DUMP_LIMIT_BYTES:
        raise ValueError("invalid linked output dump byte limit")
    wanted = {(row["program"], row["output"]): row for row in selected}
    expected = {
        (row["program"], row["step"], row["output"]): row
        for row in digests if (row["program"], row["output"]) in wanted
    }
    if len(expected) != sum(session.programs[program].steps
                            for program, _output in wanted):
        raise ValueError("selected output dumps differ from the session ABI")
    values = {}
    for line in log.splitlines():
        if not line.startswith("SESSION_HEX "):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError("malformed session output dump")
        try:
            program, step, output, byte_count = map(int, parts[1:5])
        except ValueError as error:
            raise ValueError("malformed session output dump") from error
        key = (program, step, output)
        if (key not in expected or key in values or
                byte_count != expected[key]["bytes"] or byte_count > byte_limit or
                len(parts[5]) != byte_count * 2):
            raise ValueError("session output dump differs from its capture ABI")
        try:
            data = bytes.fromhex(parts[5])
        except ValueError as error:
            raise ValueError("malformed session output dump") from error
        if hashlib.sha256(data).hexdigest() != expected[key]["sha256"]:
            raise ValueError("session output dump differs from its guest digest")
        values[key] = data
    if set(values) != set(expected):
        raise ValueError("complete session lacks a selected output dump")
    if not values:
        return []
    dump_dir = out_dir / "output_dumps"
    dump_dir.mkdir()
    result = []
    for (program, step, output), data in sorted(values.items()):
        path = dump_dir / f"p{program}_s{step}_o{output}.bin"
        path.write_bytes(data)
        result.append({"program": program, "step": step, "output": output,
                       "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "path": str(path.relative_to(out_dir))})
    return result


def run_session(
    link_receipt: Path, out_dir: Path, *, session_root: Path,
    merlin_src: Path, riscv_root: Path, spike_extlib: Path | None,
    timeout_s: int, progress_interval_s: int = 60,
    perf_config: str | None = None, perf_set: str | None = None,
) -> dict:
    if timeout_s <= 0 or progress_interval_s <= 0:
        raise ValueError("Spike timeout and progress interval must be positive")
    if perf_set and not perf_config:
        raise ValueError("performance overrides require a model preset")
    if perf_config and spike_extlib is None:
        raise ValueError("Nicolas timing model requires a Spike extension")
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower.session_bundle import load

    session = load(session_root)
    link = json.loads(link_receipt.read_text(encoding="utf-8"))
    if link.get("schema") != "mx_gemmini.linked_merlin_session.v1":
        raise ValueError("unknown linked session receipt")
    if Path(link["session_root"]).resolve() != session_root.resolve():
        raise ValueError("linked session root differs from the requested capture")
    elf = Path(link["elf"]).resolve(strict=True)
    if elf.stat().st_size != link["elf_bytes"] or _sha(elf) != link["elf_sha256"]:
        raise ValueError("linked session ELF changed after its receipt")
    recorded_programs = [(row["name"], row["steps"], row["outputs"])
                         for row in link["programs"]]
    captured_programs = [(row.name, row.steps, len(row.outputs)) for row in session.programs]
    if recorded_programs != captured_programs:
        raise ValueError("linked program schedule differs from capture")
    layout = link.get("arena_layout")
    if layout is not None:
        nm = riscv_root / "bin/riscv64-unknown-elf-nm"
        if not nm.is_file():
            raise ValueError("Spike session needs RV64 nm to verify the linked static extent")
        static_end_symbol = link.get("static_end_symbol", "_stack_top")
        if static_end_symbol not in {"_stack_top", "_merlin_static_end"}:
            raise ValueError("unknown linked session static extent symbol")
        measured_static_end = _symbol_address(nm, elf, static_end_symbol)
        weight_bytes = link.get("shared_weight_bytes")
        validate_linked_layout(layout, mem_bytes=int(link["mem_bytes"]),
                               weight_bytes=(int(weight_bytes) if weight_bytes is not None
                                             else None),
                               measured_static_end=measured_static_end)
    if sum(row["selected_operations"] for row in link["programs"]) and spike_extlib is None:
        raise ValueError("MX session requires a pinned Spike extension")
    spike = riscv_root / "bin/spike"
    if not spike.is_file():
        raise ValueError("selected Spike binary is absent")
    if out_dir.exists():
        raise ValueError("session run output directory must not already exist")
    out_dir.mkdir(parents=True)
    isa = ("rv64gc_zicntr" if link.get("performance_schema")
           == "spike_rdcycle_rdinstret_per_step.v1" else "rv64gc")
    command = [str(spike), f"--isa={isa}"]
    if spike_extlib is not None:
        command.extend((f"--extlib={spike_extlib.resolve(strict=True)}", "--extension=gemmini"))
    command.extend((f"-m0x80000000:{hex(int(link['mem_bytes']))}", str(elf)))
    log_path = out_dir / "spike.log"
    perf_path = out_dir / "gemmini_perf.txt"
    environment = os.environ.copy()
    for name in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET",
                 "GEMMINI_PERF_OUT"):
        environment.pop(name, None)
    environment["GEMMINI_MODE"] = "both" if perf_config else "func"
    if perf_config:
        environment["GEMMINI_PERF_CONFIG"] = perf_config
        environment["GEMMINI_PERF_OUT"] = str(perf_path)
    if perf_set:
        environment["GEMMINI_PERF_SET"] = perf_set
    started = time.monotonic()
    with log_path.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment)
        try:
            while True:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    exit_code = 124
                    break
                try:
                    exit_code = process.wait(timeout=min(progress_interval_s, remaining))
                    break
                except subprocess.TimeoutExpired:
                    stream.flush()
                    progress = session_progress(
                        log_path.read_text(encoding="utf-8", errors="replace"), session,
                        elapsed_wall_s=time.monotonic() - started)
                    (out_dir / "progress.json").write_text(
                        json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    print(json.dumps({key: progress[key] for key in (
                        "elapsed_wall_s", "completed_steps", "total_steps", "active_step")}),
                        flush=True)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    log = log_path.read_text(encoding="utf-8", errors="replace")
    elapsed_wall_s = time.monotonic() - started
    (out_dir / "progress.json").write_text(json.dumps(session_progress(
        log, session, elapsed_wall_s=elapsed_wall_s), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    observation = observe_session_output(
        log, session, exit_code=exit_code,
        require_perf=link.get("performance_schema") == "spike_rdcycle_rdinstret_per_step.v1")
    output_dumps = (extract_output_dumps(
        log, session, link.get("dump_outputs", []), observation["output_digests"],
        out_dir, link.get("dump_limit_bytes", DEFAULT_DUMP_LIMIT_BYTES))
        if observation["complete"] else [])
    timing_model_active = "gemmini: GEMMINI_MODE=both" in log
    timing_report = perf_path.read_text(encoding="utf-8") if perf_path.is_file() else ""
    if perf_config and not timing_model_active:
        raise ValueError("selected Spike extension did not enable Nicolas's timing model")
    if (perf_config and observation["complete"] and
            f"gemmini perf [{perf_config}]" not in timing_report):
        raise ValueError("complete session lacks the selected timing-model report")
    receipt = {
        "schema": "mx_gemmini.merlin_session_spike_run.v1",
        "status": ("complete_numeric_qualification_pending" if observation["complete"]
                   else "incomplete"),
        "link_receipt_sha256": _sha(link_receipt),
        "spike_sha256": _sha(spike),
        "extension_sha256": _sha(spike_extlib) if spike_extlib else None,
        "command": command, "exit_code": exit_code,
        "performance_schema": link.get("performance_schema"),
        "timing_mode": environment["GEMMINI_MODE"],
        "timing_config": perf_config,
        "timing_overrides": perf_set,
        "timing_model_active": timing_model_active,
        "timing_report_sha256": _sha(perf_path) if perf_path.is_file() else None,
        "elapsed_wall_s": elapsed_wall_s,
        "log_sha256": _sha(log_path), **observation,
        "output_dumps": output_dumps,
    }
    (out_dir / "mx_session_run.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-receipt", required=True, type=Path)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--merlin-src", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--spike-extlib", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--progress-interval-s", type=int, default=60)
    parser.add_argument("--perf-config", choices=("mx_rocket", "e4m3_vpu", "firesim"))
    parser.add_argument("--perf-set", help="comma-separated Nicolas model section.name=value overrides")
    args = parser.parse_args()
    receipt = run_session(
        args.link_receipt.resolve(strict=True), args.out_dir.absolute(),
        session_root=args.session_root.resolve(strict=True),
        merlin_src=args.merlin_src.resolve(strict=True),
        riscv_root=args.riscv_root.resolve(strict=True),
        spike_extlib=args.spike_extlib.resolve(strict=True) if args.spike_extlib else None,
        timeout_s=args.timeout_s, progress_interval_s=args.progress_interval_s,
        perf_config=args.perf_config, perf_set=args.perf_set)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
