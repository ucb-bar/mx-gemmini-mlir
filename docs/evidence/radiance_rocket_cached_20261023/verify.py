"""Recompute the cached comparison from complete recorded output streams."""

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from support import gates, parse_console, quality


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    require(sha((root / "support.py").read_bytes()) == manifest["support_sha256"],
            "frozen comparison helper changed")
    raw = (root / "observations.tar.xz").read_bytes()
    require(sha(raw) == manifest["archive_sha256"], "archive changed")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for entry in archive:
            require(entry.isfile() and entry.name not in files, "invalid archive member")
            files[entry.name] = archive.extractfile(entry).read()
    require(set(files) == set(manifest["members"]), "archive roster changed")
    for name, data in files.items():
        require({"bytes": len(data), "sha256": sha(data)} == manifest["members"][name],
                "archive member changed: " + name)
    report = json.loads(files["checks/full_cached_comparison.json"])
    source = json.loads(files["source/receipt.json"])
    run = json.loads(files["compiler/mx_session_run.json"])
    link_raw = files["compiler/mx_session_link.json"]
    require(sha(files["source/receipt.json"]) == report["port_receipt_sha256"],
            "source receipt identity differs")
    require(run["link_receipt_sha256"] == sha(link_raw), "compiler link identity differs")
    require(sha(files["compiler/mx_session_run.json"])
            == report["compiler_state_audit"]["run_receipt_sha256"],
            "compiler run identity differs from original audit")
    require(sha(files["reference/prefill.npy"]) == source["capture_files_sha256"]["golden.npy"]
            and sha(files["reference/decode.npz"])
            == source["session_capture_sha256"]["stages/decode/session_quality_fp32.npz"],
            "floating references differ from source capture")
    require(run["complete"] is True and run["exit_code"] == 0, "compiler incomplete")
    for name in ("spike.log", "perf.txt", "phases.bin", "token_ids.bin", "checks.json"):
        require(sha(files["source/" + name]) == source["files_sha256"][name],
                "source artifact identity differs: " + name)
    console = files["source/spike.log"].decode()
    cycles, _, outputs = parse_console(
        console, json.loads(files["source/checks.json"]), source["phase_count"], False
    )
    steps = [tuple(map(int, row)) for row in re.findall(
        r"^CACHED_STEP (\d+) (\d+) ([1-9][0-9]*)$", console, re.MULTILINE)]
    reset = re.findall(r"^CACHED_RESET ([1-9][0-9]*)$", console, re.MULTILINE)
    require(len(reset) == 1 and [(s, p) for s, p, _ in steps]
            == [(s, 128 + s) for s in range(33)], "source positions incomplete")
    require(cycles == int(reset[0]) + sum(t for _, _, t in steps)
            == source["cycles"] == report["source_cycles"], "source timing differs")
    require(steps[0][2] == report["source_prefill_cycles"]
            and sum(t for _, _, t in steps[1:]) == report["source_decode_cycles"],
            "source stage timing differs")
    for index in (0, 1):
        require(outputs[index] == files[f"source/observed{index}.bin"],
                "source output differs from console")
    state = np.frombuffer(outputs[1], "<f4")
    require(state.size == report["source_final_kv_words_checked_finite"]
            and np.isfinite(state).all(), "source final cache invalid")
    compiler_console = files["compiler/spike.log"]
    require(sha(compiler_console) == run["log_sha256"], "compiler console changed")
    require(compiler_console.decode().splitlines().count("DONE") == 1,
            "compiler terminal completion missing")
    dumped = {}
    for line in compiler_console.decode().splitlines():
        if line.startswith("SESSION_HEX "):
            _, p, s, o, n, data = line.split()
            key = int(p), int(s), int(o)
            require(key not in dumped, "duplicated compiler output")
            dumped[key] = bytes.fromhex(data)
            require(len(dumped[key]) == int(n), "incomplete compiler console output")
    keys = [(0, 0, 0)] + [(1, s, 0) for s in range(32)]
    require(set(dumped) == set(keys), "compiler vocabulary outputs incomplete")
    require(len(run["output_dumps"]) == 33, "compiler dump roster differs")
    for row in run["output_dumps"]:
        data = files["compiler/" + row["path"]]
        key = row["program"], row["step"], row["output"]
        require(data == dumped[key] and sha(data) == row["sha256"]
                and len(data) == row["bytes"], "compiler output changed")
    require(run["execution_total_performance"]["cycles"] == report["compiler_cycles"],
            "compiler inclusive timing differs")
    reference = [np.load(io.BytesIO(files["reference/prefill.npy"]), allow_pickle=False)]
    with np.load(io.BytesIO(files["reference/decode.npz"]), allow_pickle=False) as data:
        reference.extend(data["logits"])
    reference = np.stack(reference).reshape(33, 32000)
    source_quality = quality(np.frombuffer(outputs[0], "<f4").reshape(33, 32000), reference)
    compiler_quality = quality(np.stack([np.frombuffer(dumped[k], "<f4") for k in keys]),
                               reference)
    require(source_quality == report["source_quality"], "source numerical metrics differ")
    require(compiler_quality == report["compiler_quality"], "compiler numerical metrics differ")
    acceptance = gates(compiler_quality, source_quality)
    require(acceptance == report["numerical_gates"], "numerical gates differ")
    require(not all(acceptance.values()) and report["comparison_qualified"] is False
            and report["qualified_compiler_speedup"] is None, "qualification differs")
    print(json.dumps({"recorded_outputs_and_timing_verified": True,
                      "source_cycles": cycles, "compiler_cycles": report["compiler_cycles"],
                      "numerical_gates": acceptance, "comparison_qualified": False}, indent=2))


if __name__ == "__main__":
    main()
