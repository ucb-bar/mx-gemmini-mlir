"""Run a linked MX/Merlin session on pinned Spike and verify its result ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


_BYTES = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2, "f16": 2, "i32": 4}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def observe_session_output(log: str, session, *, exit_code: int) -> dict:
    """Accept only a correct prefix of the captured session's output/state ABI."""
    expected = _expected(session)
    seen = 0
    records = []
    for raw in log.splitlines(keepends=True):
        if exit_code != 0 and not raw.endswith("\n"):
            break  # A timed-out Spike process may stop between HTIF characters.
        line = raw.rstrip("\n")
        parts = line.split()
        if not parts or parts[0] not in {"SESSION_BEGIN", "SESSION_SHA"}:
            continue
        if seen >= len(expected):
            raise ValueError("session emitted more result records than its capture ABI")
        kind, program, step, output, bytes_expected = expected[seen]
        if parts[0] == "SESSION_BEGIN":
            if len(parts) != 3 or kind != "begin" or [int(x) for x in parts[1:]] != [program, step]:
                raise ValueError("session program/step order differs from capture ABI")
        else:
            if (len(parts) != 6 or kind != "sha" or
                    [int(x) for x in parts[1:5]] != [program, step, output, bytes_expected]
                    or not _SHA.fullmatch(parts[5])):
                raise ValueError("session result digest differs from capture ABI")
            records.append({"program": program, "step": step, "output": output,
                            "bytes": bytes_expected, "sha256": parts[5]})
        seen += 1
    done = "DONE" in log.splitlines()
    metrics = [line for line in log.splitlines()
               if line.startswith("METRIC memref_rank_mismatch ")]
    guest_failure = any(line.startswith("FATAL ") or "*** FAILED" in line
                        for line in log.splitlines())
    complete = not guest_failure and exit_code == 0 and done and seen == len(expected) and metrics == [
        "METRIC memref_rank_mismatch 0"]
    if exit_code == 0 and not complete:
        raise ValueError("Spike exited successfully without a complete valid session")
    if done and not complete:
        raise ValueError("session printed DONE with incomplete or invalid result ABI")
    return {"complete": complete, "observed_events": seen,
            "expected_events": len(expected), "output_digests": records}


def run_session(
    link_receipt: Path, out_dir: Path, *, session_root: Path,
    merlin_src: Path, riscv_root: Path, spike_extlib: Path | None,
    timeout_s: int,
) -> dict:
    if timeout_s <= 0:
        raise ValueError("Spike timeout must be positive")
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
        arena_start = int(layout["arena_base"])
        arena_end = arena_start + int(layout["arena_bytes"])
        weight_start = int(layout["weights_base"])
        weight_bytes = link.get("shared_weight_bytes")
        weight_end = weight_start + int(weight_bytes) if weight_bytes is not None else weight_start
        if (arena_start < 0x80000000 or arena_end > 0x80000000 + int(link["mem_bytes"])
                or max(arena_start, weight_start) < min(arena_end, weight_end)
                or int(layout["mem_bytes"]) > int(link["mem_bytes"])):
            raise ValueError("linked session arena overlaps weights or exceeds Spike memory")
    if sum(row["selected_operations"] for row in link["programs"]) and spike_extlib is None:
        raise ValueError("MX session requires a pinned Spike extension")
    spike = riscv_root / "bin/spike"
    if not spike.is_file():
        raise ValueError("selected Spike binary is absent")
    if out_dir.exists():
        raise ValueError("session run output directory must not already exist")
    out_dir.mkdir(parents=True)
    command = [str(spike), "--isa=rv64gc"]
    if spike_extlib is not None:
        command.extend((f"--extlib={spike_extlib.resolve(strict=True)}", "--extension=gemmini"))
    command.extend((f"-m0x80000000:{hex(int(link['mem_bytes']))}", str(elf)))
    log_path = out_dir / "spike.log"
    with log_path.open("wb") as stream:
        try:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout_s, check=False)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    log = log_path.read_text(encoding="utf-8", errors="replace")
    observation = observe_session_output(log, session, exit_code=exit_code)
    receipt = {
        "schema": "mx_gemmini.merlin_session_spike_run.v1",
        "status": ("complete_numeric_qualification_pending" if observation["complete"]
                   else "incomplete"),
        "link_receipt_sha256": _sha(link_receipt),
        "spike_sha256": _sha(spike),
        "extension_sha256": _sha(spike_extlib) if spike_extlib else None,
        "command": command, "exit_code": exit_code,
        "log_sha256": _sha(log_path), **observation,
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
    args = parser.parse_args()
    receipt = run_session(
        args.link_receipt.resolve(strict=True), args.out_dir.absolute(),
        session_root=args.session_root.resolve(strict=True),
        merlin_src=args.merlin_src.resolve(strict=True),
        riscv_root=args.riscv_root.resolve(strict=True),
        spike_extlib=args.spike_extlib.resolve(strict=True) if args.spike_extlib else None,
        timeout_s=args.timeout_s)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
