"""Verify the corrected full SmolVLA run and its FP32 comparison."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "smolvla_full_action_values_20261014"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    delta = lhs - rhs
    norm = max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300)
    ref_rms = max(float(np.sqrt(np.mean(rhs * rhs))), 1e-300)
    return {"mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "normalized_rmse": float(np.sqrt(np.mean(delta * delta))) / ref_rms,
            "max_abs": float(np.max(np.abs(delta))),
            "cosine": float(np.clip(np.sum(lhs * rhs) / norm, -1.0, 1.0))}


def verify() -> dict:
    receipt = json.loads((HERE / "receipt.json").read_text())
    if (receipt.get("schema") != "mx_gemmini.smolvla_scalar_full_action.v1"
            or receipt.get("numerical_qualification") != "not_established"
            or receipt.get("cycle_ratio_valid") is not False):
        raise ValueError("unexpected full action archive schema")
    for name, digest in receipt["artifact_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"changed artifact: {name}")
    run = json.loads((HERE / "run.json").read_text())
    link = json.loads((HERE / "link.json").read_text())
    old_run = json.loads((PRIOR / "tap_run.json").read_text())
    old_link = json.loads((PRIOR / "tap_link.json").read_text())
    report = json.loads((HERE / "fp32_comparison.json").read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run["link_receipt_sha256"] != sha(HERE / "link.json")
            or run["timing_config"] != "e4m3_vpu"
            or run["timing_overrides"] != "mem.host_tracking=0"
            or report["run_receipt_sha256"] != sha(HERE / "run.json")
            or report["link_receipt_sha256"] != sha(HERE / "link.json")
            or report["reference_sha256"] != sha(PRIOR / "session_goldens.npz")):
        raise ValueError("full run and FP32 comparison identities differ")
    for name in ("prefix_encode", "action_decode"):
        if (old_link["stage_object_sha256"][name]
                != link["stage_object_sha256"][name]):
            raise ValueError(f"unrelated stage object changed: {name}")
    if (old_link["shared_weight_sha256"] != link["shared_weight_sha256"]
            or old_link["profile_sha256"] != link["profile_sha256"]
            or old_link["programs"] != link["programs"]
            or old_link["stage_object_sha256"]["flow_denoise"]
            == link["stage_object_sha256"]["flow_denoise"]):
        raise ValueError("corrected flow stage is not the only changed stage")
    log = gzip.decompress((HERE / "spike.log.gz").read_bytes())
    if (hashlib.sha256(log).hexdigest() != run["log_sha256"]
            or b"DONE\n" not in log
            or sha(HERE / "gemmini_perf.txt") != run["timing_report_sha256"]):
        raise ValueError("Spike log or performance report differs")
    selected = {(row["program"], row["step"], row["output"]): row
                for row in run["output_dumps"]}
    expected_keys = ({(1, step, output) for step in range(10)
                      for output in (0, 2)} | {(2, 0, 0)})
    if set(selected) != expected_keys:
        raise ValueError("full flow/action/timestep output set is incomplete")
    with np.load(PRIOR / "session_goldens.npz", allow_pickle=False) as data:
        expected = data["actions"]
    if expected.shape != (10, 1, 50, 32):
        raise ValueError("FP32 reference shape changed")
    time = np.float32(1.0)
    values = []
    for step in range(10):
        row = selected[1, step, 0]
        raw = HERE / row["path"]
        if raw.stat().st_size != row["bytes"] or sha(raw) != row["sha256"]:
            raise ValueError(f"flow step {step} output differs")
        data = np.fromfile(raw, dtype="<f4").reshape(1, 50, 32)
        values.append(data)
        time = np.float32(time + np.float32(-0.1))
        row = selected[1, step, 2]
        raw = HERE / row["path"]
        if (raw.stat().st_size != row["bytes"] or sha(raw) != row["sha256"]
                or struct.unpack("<f", raw.read_bytes())[0] != float(time)):
            raise ValueError(f"flow step {step} timestep differs from FP32")
        observed = metrics(data, expected[step])
        stored = report["flow_step_metrics"][step]
        if any(abs(observed[key] - stored[key]) > 1e-10 for key in observed):
            raise ValueError(f"flow step {step} FP32 metrics differ")
    row = selected[2, 0, 0]
    action = HERE / row["path"]
    if (action.stat().st_size != row["bytes"] or sha(action) != row["sha256"]
            or action.read_bytes() != (HERE / selected[1, 9, 0]["path"]).read_bytes()
            or report["final_action_matches_flow_bytes"] is not True):
        raise ValueError("final action does not equal the last flow output")
    if (old_run["output_digests"] == run["output_digests"]
            or report["final_action_metrics"] != report["flow_step_metrics"][-1]):
        raise ValueError("corrected run did not change outputs as expected")
    old_steps = {(row["program"], row["step"]): row["cycles"]
                 for row in old_run["step_performance"]}
    new_steps = {(row["program"], row["step"]): row["cycles"]
                 for row in run["step_performance"]}
    required = {(0, 0), (2, 0)} | {(1, step) for step in range(10)}
    if (set(old_steps) != required or set(new_steps) != required
            or any(abs(old_steps[key] - new_steps[key]) > 1
                   for key in ((0, 0), (2, 0)))
            or any(new_steps[1, step] >= old_steps[1, step]
                   for step in range(10))):
        raise ValueError("matched output-tap stage timing differs unexpectedly")
    return {"old_final_cosine": json.loads((PRIOR / "fp32_comparison.json").read_text())
            ["final_action_metrics"]["cosine"],
            "new_final_cosine": report["final_action_metrics"]["cosine"],
            "new_final_normalized_rmse": report["final_action_metrics"]["normalized_rmse"],
            "new_cycles": run["total_performance"]["cycles"],
            "old_flow_step0_cycles": old_steps[1, 0],
            "new_flow_step0_cycles": new_steps[1, 0]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
