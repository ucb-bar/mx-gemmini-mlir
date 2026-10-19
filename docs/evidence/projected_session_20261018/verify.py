"""Verify every fixture logit/state word, compact weights, admission and timed phases."""

import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def weights(path):
    raw = path.read_bytes()
    size, = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    values = {}
    for name, row in header.items():
        if name == "__metadata__":
            continue
        assert row["dtype"] == "F32"
        begin, end = row["data_offsets"]
        values[name] = np.frombuffer(payload[begin:end], dtype="<f4").reshape(row["shape"])
    return values, payload


def main():
    here = Path(__file__).resolve().parent
    for name, expected in json.loads((here / "artifacts.json").read_text()).items():
        assert digest((here / name).read_bytes()) == expected, name
    capture = here / "capture"
    w, original_blob = weights(capture / "stages/prefill/weights.safetensors")
    assert np.array_equal(w["linear.weight"], np.eye(32, dtype=np.float32))
    assert np.array_equal(w["linear.bias"], np.full(32, 0.125, np.float32))
    with np.load(capture / "stages/prefill/inputs.npz", allow_pickle=False) as data:
        x, state = data["in0"], data["in1"].copy()
    with np.load(capture / "stages/decode/inputs.npz", allow_pickle=False) as data:
        assert np.all(data["in1"] == -7)
    with np.load(capture / "stages/decode/session_inputs.npz", allow_pickle=False) as data:
        frames = data["frames"]
    assert x.shape == state.shape == (16, 32) and len(frames) == 3
    expected = {}
    for program, step, frame in [(0, 0, x), *[(1, i, value) for i, value in enumerate(frames)]]:
        value = frame @ w["linear.weight"].T
        value = (value + w["linear.bias"]) + state
        state = value + np.float32(0.5)
        expected[program, step, 0] = value.copy()
        expected[program, step, 1] = state.copy()
    assert np.unique(expected[1, 2, 0]).size > 4
    summaries = {}
    for mode in ("original", "projected"):
        directory = here / mode
        link_bytes = (directory / "mx_session_link.json").read_bytes()
        link = json.loads(link_bytes)
        run = json.loads((directory / "mx_session_run.json").read_text())
        assert run["complete"] and run["exit_code"] == 0
        assert run["link_receipt_sha256"] == digest(link_bytes)
        assert run["timing_mode"] == "both" and run["timing_config"] == "e4m3_vpu"
        assert run["timing_overrides"] == "mem.host_tracking=0"
        assert run["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
        console = gzip.decompress((directory / "spike.log.gz").read_bytes())
        assert digest(console) == run["log_sha256"]
        assert console.decode().splitlines().count("DONE") == 1
        observed = {}
        for line in console.decode().splitlines():
            parts = line.split()
            if parts and parts[0] == "SESSION_HEX":
                assert len(parts) == 6
                key = tuple(map(int, parts[1:4]))
                assert key in expected and key not in observed
                raw = bytes.fromhex(parts[5])
                assert len(raw) == int(parts[4]) == expected[key].nbytes
                actual = np.frombuffer(raw, dtype="<f4").reshape(expected[key].shape)
                assert np.array_equal(actual.view(np.uint32), expected[key].view(np.uint32)), (mode, key)
                observed[key] = digest(raw)
        assert observed.keys() == expected.keys()
        for row in run["output_digests"]:
            assert row["sha256"] == observed[row["program"], row["step"], row["output"]]
        for key in ("cycles", "instructions"):
            steps = sum(row[key] for row in run["step_performance"])
            setup = sum(row[key] for row in run["preparation_performance"])
            assert steps == run["total_performance"][key]
            assert setup == run["preparation_total_performance"][key]
            assert steps + setup == run["execution_total_performance"][key]
        assert [row["program"] for row in run["preparation_performance"]] == [-1, 0, 1]
        assert "console_and_phase_bookkeeping_excluded" in run["execution_timing_scope"]
        elf = gzip.decompress((directory / "session.elf.gz").read_bytes())
        assert len(elf) == link["elf_bytes"] and digest(elf) == link["elf_sha256"]
        admission = (directory / "admission.json").read_bytes()
        assert digest(admission) == link["linked_elf_admission"]["report_sha256"]
        assert json.loads(admission)["elf_sha256"] == digest(elf)
        assert json.loads(admission)["status"] == "accepted"
        assert link["linked_elf_admission_required"]
        blob = (directory / "weights.bin").read_bytes()
        assert digest(blob) == link["shared_weight_sha256"] and len(blob) == link["shared_weight_bytes"]
        assert blob == (original_blob if mode == "original" else w["linear.bias"].tobytes())
        for name in ("prefill", "decode"):
            obj = gzip.decompress((directory / name / "model.o.gz").read_bytes())
            assert digest(obj) == link["stage_object_sha256"][name]["model.o"]
            if mode == "projected":
                proof = json.loads((directory / name / "entry_weight_projection.json").read_text())
                assert proof["removed_original_indices"] == [0]
                assert proof["original_to_projected"] == [None, 0, 1, 2]
                assert link["entry_projections"][name]["argument_table_sha256"] == proof["argument_table_sha256"]
                assert link["entry_projections"][name]["original_to_projected"] == proof["original_to_projected"]
        summaries[mode] = {"step_cycles": run["total_performance"]["cycles"],
                           "preparation_cycles": run["preparation_total_performance"]["cycles"],
                           "accounted_cycles": run["execution_total_performance"]["cycles"],
                           "shared_weight_bytes": len(blob), "verified_output_words": 4096}
    print(json.dumps({"scope": "small_recurrent_fixture_not_full_model", "verified": summaries}, indent=2))


if __name__ == "__main__":
    main()
