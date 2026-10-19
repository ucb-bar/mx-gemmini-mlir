"""Verify archived output bits, scalar BF16 arithmetic, traffic and receipts."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def round_bf16(values):
    words = np.asarray(values, dtype="<f4").view("<u4")
    return ((words + 0x7fff + ((words >> 16) & 1)) & np.uint32(0xffff0000)).view("<f4")


def main():
    manifest = load("manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    stage_outputs, cycles = [], []
    for name in ("materialized", "connected"):
        directory = ROOT / "stage" / name
        receipt = json.loads((directory / "mx_stage_probe.json").read_text())
        assert receipt["spike"]["done"]
        assert receipt["elf_admission_selection"] == "structural_digest_and_declared_isa"
        log = gzip.decompress((directory / "spike.log.gz").read_bytes()).decode()
        line, = [line for line in log.splitlines() if line.startswith("OUT ")]
        words = line.split()
        values = np.asarray([int(word) for word in words[2:]], dtype="<u4")
        assert len(values) == int(words[1]) == 1024
        digest_line, = [line for line in log.splitlines() if line.startswith("OUT_SHA256 ")]
        assert sha(values.tobytes()) == digest_line.split()[-1]
        stage_outputs.append(values)
        cycle_line, = [line for line in log.splitlines() if line.startswith("METRIC cycles ")]
        cycles.append(int(cycle_line.split()[-1]))
    assert np.array_equal(*stage_outputs)
    assert cycles[1] < cycles[0]
    planned = load("stage/connected/mx_catalog.json")["vpu_elementwise"]["connected_multiplies"]
    realized = load("stage/connected/mx_vpu_elementwise_rewrite.json")["connected_multiplies"]
    assert realized == [{key: row[key] for key in ("regions", "shape", "symbol")} for row in planned]
    assert len(realized) == 1
    previous = ROOT.parent / "connected_model_section_20261016"
    for case in ("tiny_mlp", "tail137", "fallback137"):
        report = load(case + "/receipt.json")
        payload = gzip.decompress((previous / case / "payload.bin.gz").read_bytes())
        assert sha(payload) == report["source_files_sha256"]["payload.bin"]
        arrays = np.frombuffer(payload, dtype="<f4").reshape(4, -1)
        with np.errstate(over="ignore", invalid="ignore"):
            golden = round_bf16(round_bf16(round_bf16(arrays[0]) * round_bf16(arrays[1])) *
                                round_bf16(arrays[2]))
        assert np.array_equal(golden.view("<u4"), arrays[3].view("<u4")), case
        digest = 2166136261
        for word in golden.view("<u4"):
            digest = ((digest ^ int(word)) * 16777619) & 0xffffffff
        a, b = (report["results"][name] for name in ("materialized", "connected"))
        for name, result in report["results"].items():
            log = (ROOT / case / name / "spike.log").read_text()
            actual = [dict(word.split("=", 1) for word in line.split()[1:])
                      for line in log.splitlines() if line.startswith("SECTION ")]
            assert actual == result["checks"] and len(actual) == 3
            assert all(int(row["mismatches"]) == 0 and int(row["hash"]) == digest for row in actual)
        saved = min(int(r["cycles"]) for r in a["checks"][1:]) - max(int(r["cycles"]) for r in b["checks"][1:])
        assert saved == report["warm_saved_cycles"]
        assert (a["counters"]["load_bytes"] - b["counters"]["load_bytes"] ==
                report["removed_transfer_bytes_per_call"] // 2 * 3)
        assert (saved < 0) == report["expected_fallback"]
    for name in ("prefill", "decode"):
        pairs = load(name + "_rewrite.json")["connected_multiplies"]
        assert len(pairs) == 22
        assert all(
            value == "host" for value in load(name + "_policy.json")["vpu_placements"]["softmax"].values())
    print(json.dumps({"verified": True, "stage_cycles": cycles,
                      "scope": "paired compiler execution and section arithmetic; no full-model speedup"}))


if __name__ == "__main__":
    main()
