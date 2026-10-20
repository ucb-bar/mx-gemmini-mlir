"""Check session state invariants directly from recorded simulator output.

Uses Python and NumPy only. These checks supplement numerical comparison:
correct logits alone do not establish that positions, immutable prefix caches
and recurrent timesteps were handed between stages correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_state(kind, rows):
    """Derive expected state from the model's session semantics."""
    selected = {(r["program"], r["step"], r["output"]): r for r in rows}
    if len(selected) != len(rows):
        raise ValueError("duplicated recurrent output")
    if kind == "tinyllama":
        positions = [(0, 0, 128)] + [(1, step, 129 + step) for step in range(32)]
        wanted = {(p, s, o) for p, s, _ in positions for o in range(3)}
        if set(selected) != wanted:
            raise ValueError("TinyLlama needs all 33 outputs and two states")
        caches = []
        for program, step, position in positions:
            if selected[program, step, 0]["bytes"] != 128000:
                raise ValueError("incomplete TinyLlama vocabulary output")
            row = selected[program, step, 2]
            expected = np.full(22, position, dtype="<i8").tobytes()
            if row["bytes"] != len(expected) or row["sha256"] != digest(expected):
                raise ValueError(
                    f"incorrect per-layer cache position at {(program, step)}"
                )
            state = selected[program, step, 1]
            if state["bytes"] != 2 * 22 * 4 * 160 * 64 * 4:
                raise ValueError("incomplete TinyLlama K/V state")
            caches.append(state["sha256"])
        if len(set(caches)) != 33:
            raise ValueError("TinyLlama K/V state did not change at every append")
        return {
            "position_vectors_checked": 33,
            "distinct_complete_kv_states": 33,
            "final_position_per_layer": 160,
        }
    if kind != "smolvla":
        raise ValueError("unsupported session kind")
    wanted = (
        {(0, 0, o) for o in range(2)}
        | {(1, s, o) for s in range(10) for o in range(3)}
        | {(2, 0, 0)}
    )
    if set(selected) != wanted:
        raise ValueError("SmolVLA needs prefix, ten flow steps and decoded action")
    prefix = selected[0, 0, 1]
    if prefix["bytes"] != 2314240:
        raise ValueError("incomplete SmolVLA prefix K/V state")
    timestep = np.float32(1)
    timestep_words = []
    actions = []
    for step in range(10):
        cache = selected[1, step, 1]
        if (cache["bytes"], cache["sha256"]) != (prefix["bytes"], prefix["sha256"]):
            raise ValueError(f"immutable prefix cache changed at flow step {step}")
        timestep = np.float32(timestep + np.float32(-0.1))
        raw = np.asarray([timestep], dtype="<f4").tobytes()
        row = selected[1, step, 2]
        if row["bytes"] != 4 or row["sha256"] != digest(raw):
            raise ValueError(f"incorrect recurrent timestep at flow step {step}")
        timestep_words.append(raw.hex())
        action = selected[1, step, 0]
        if action["bytes"] != 6400:
            raise ValueError("incomplete recurrent action tensor")
        actions.append(action["sha256"])
    if len(set(actions)) != 10:
        raise ValueError("flow state did not change at every denoising step")
    final = selected[2, 0, 0]
    if final["bytes"] != 6400 or final["sha256"] != actions[-1]:
        raise ValueError("decoded action differs from the final flow state")
    return {
        "immutable_prefix_cache_handoffs_checked": 10,
        "timestep_words_little_endian": timestep_words,
        "distinct_complete_flow_states": 10,
        "decoded_action_matches_final_flow": True,
    }


def audit(kind, run_path, link_path):
    run_raw, link_raw = run_path.read_bytes(), link_path.read_bytes()
    run, link = json.loads(run_raw), json.loads(link_raw)
    if (
        run.get("complete") is not True
        or run.get("exit_code") != 0
        or run["link_receipt_sha256"] != digest(link_raw)
    ):
        raise ValueError("session did not complete or link identity changed")
    console_raw = (run_path.parent / "spike.log").read_bytes()
    if digest(console_raw) != run["log_sha256"]:
        raise ValueError("console bytes differ from the run receipt")
    console = console_raw.decode()
    if console.splitlines().count("DONE") != 1 or re.search(
        r"DEADLOCK|^FATAL|^FAIL ", console, re.MULTILINE
    ):
        raise ValueError("missing terminal completion or simulator failure")
    rows = []
    for line in console.splitlines():
        if line.startswith("SESSION_SHA "):
            _, p, s, o, n, sha = line.split()
            if re.fullmatch("[0-9a-f]{64}", sha) is None:
                raise ValueError("invalid simulator state digest")
            rows.append(
                {
                    "program": int(p),
                    "step": int(s),
                    "output": int(o),
                    "bytes": int(n),
                    "sha256": sha,
                }
            )
    if rows != run["output_digests"]:
        raise ValueError("console state hashes differ from the receipt")
    invariants = check_state(kind, rows)
    dumped = {}
    for line in console.splitlines():
        if line.startswith("SESSION_HEX "):
            _, p, s, o, n, encoded = line.split()
            key = int(p), int(s), int(o)
            if key in dumped:
                raise ValueError("duplicated complete output")
            raw = bytes.fromhex(encoded)
            if len(raw) != int(n):
                raise ValueError("incomplete console output")
            dumped[key] = raw
    expected_dump_keys = (
        {(0, 0, 0)} | {(1, s, 0) for s in range(32)}
        if kind == "tinyllama"
        else {(1, s, 0) for s in range(10)} | {(2, 0, 0)}
    )
    if set(dumped) != expected_dump_keys or len(run["output_dumps"]) != len(dumped):
        raise ValueError("missing complete recurrent outputs")
    declared = {(r["program"], r["step"], r["output"]): r for r in rows}
    seen = set()
    for row in run["output_dumps"]:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("output path escapes the run")
        key = row["program"], row["step"], row["output"]
        if key in seen or key not in dumped:
            raise ValueError("duplicated or unexpected output dump")
        seen.add(key)
        raw = (run_path.parent / str(path)).read_bytes()
        if (
            raw != dumped[key]
            or len(raw) != row["bytes"]
            or digest(raw) != row["sha256"]
        ):
            raise ValueError("output bytes differ from the console or receipt")
        if digest(raw) != declared[key]["sha256"] or len(raw) != declared[key]["bytes"]:
            raise ValueError("complete output differs from its simulator digest")
        if not np.isfinite(np.frombuffer(raw, "<f4")).all():
            raise ValueError("nonfinite model output")
    image = Path(link["elf"])
    image_available = image.is_file()
    if image_available and (
        image.stat().st_size != link["elf_bytes"]
        or file_digest(image) != link["elf_sha256"]
    ):
        raise ValueError("actual final ELF differs from the measurement")
    return {
        "kind": kind,
        "run_receipt_sha256": digest(run_raw),
        "link_receipt_sha256": digest(link_raw),
        "state_invariants": invariants,
        "complete_output_tensors_checked": len(dumped),
        "measured_cycles": run["execution_total_performance"]["cycles"],
        "actual_final_elf_reopened": image_available,
        "elf_sha256": link["elf_sha256"],
        "state_checks_passed": True,
        "independent_radiance_baseline": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("tinyllama", "smolvla"), required=True)
    for name in ("run-receipt", "link-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be fresh")
    report = audit(args.kind, args.run_receipt, args.link_receipt)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
