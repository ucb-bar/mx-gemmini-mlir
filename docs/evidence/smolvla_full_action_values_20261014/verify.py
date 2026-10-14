"""Verify the archived full SmolVLA value run and its FP32 diagnostic."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FAST = HERE.parent / "nicolas_matrix_substitution_smolvla_full_20261013"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    assert actual.shape == expected.shape == (1, 50, 32)
    assert actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    difference = lhs - rhs
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(rhs * rhs)))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "normalized_rmse": rmse / max(reference_rms, 1e-300),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.clip(
            np.sum(lhs * rhs) / max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300),
            -1.0, 1.0)),
    }


def verify() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.smolvla_full_action_values.v1"
    assert receipt["numerical_qualification"] == "not_established"
    assert receipt["output_digests_match_fast_compiler"] is True
    assert receipt["cycle_ratio_valid"] is False
    for name, expected in receipt["artifact_sha256"].items():
        assert sha(HERE / name) == expected, name

    tap = json.loads((HERE / "tap_run.json").read_text())
    old = json.loads((HERE / "old_run.json").read_text())
    tap_link = json.loads((HERE / "tap_link.json").read_text())
    old_link = json.loads((HERE / "old_link.json").read_text())
    fast = json.loads((FAST / "compiler/mx_session_run.json").read_text())
    fast_link = json.loads((FAST / "compiler_link.json").read_text())
    assert sha(FAST / "compiler/mx_session_run.json") == receipt["fast_run_sha256"]
    assert sha(FAST / "compiler_link.json") == receipt["fast_link_sha256"]
    assert tap["link_receipt_sha256"] == sha(HERE / "tap_link.json")
    assert old["link_receipt_sha256"] == sha(HERE / "old_link.json")
    assert all(run["status"] == "complete_numeric_qualification_pending"
               and run["complete"] is True and run["expected_events"] ==
               run["observed_events"] == 45 and run["exit_code"] == 0
               and run["timing_mode"] == "both"
               and run["timing_config"] == "e4m3_vpu"
               and run["timing_overrides"] == "mem.host_tracking=0"
               for run in (tap, old, fast))
    assert tap["output_digests"] == old["output_digests"] == fast["output_digests"]
    assert len(tap["output_digests"]) == 33
    assert tap_link["stage_object_sha256"] == old_link["stage_object_sha256"]
    assert (tap_link["shared_weight_sha256"]
            == old_link["shared_weight_sha256"]
            == fast_link["shared_weight_sha256"])
    assert [row["capture_model_sha256"] for row in tap_link["programs"]] == [
        row["capture_model_sha256"] for row in fast_link["programs"]]
    assert tap["total_performance"]["cycles"] == receipt["tap_cycles"]
    assert fast["total_performance"]["cycles"] == receipt["fast_compiler_cycles"]
    assert receipt["tap_cycles"] != receipt["fast_compiler_cycles"]
    log = gzip.decompress((HERE / "tap_spike.log.gz").read_bytes())
    assert hashlib.sha256(log).hexdigest() == tap["log_sha256"]
    assert b"DONE\n" in log

    report = json.loads((HERE / "fp32_comparison.json").read_text())
    assert report["schema"] == "mx_gemmini.smolvla_session_fp32_comparison.v1"
    assert report["numerical_qualification"] == "not_established"
    assert report["run_receipt_sha256"] == sha(HERE / "tap_run.json")
    assert report["link_receipt_sha256"] == sha(HERE / "tap_link.json")
    assert report["reference_sha256"] == sha(HERE / "session_goldens.npz")
    assert report["final_action_matches_flow_bytes"] is True
    with np.load(HERE / "session_goldens.npz", allow_pickle=False) as data:
        expected = data["actions"]
    assert expected.shape == (10, 1, 50, 32) and expected.dtype == np.float32
    selected = {(row["program"], row["step"], row["output"]): row
                for row in tap["output_dumps"]}
    assert set(selected) == {(1, step, 0) for step in range(10)} | {(2, 0, 0)}
    values = {}
    for key, row in selected.items():
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = HERE / relative
        assert row["bytes"] == path.stat().st_size == 6400
        assert sha(path) == row["sha256"]
        values[key] = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 50, 32).copy()
    assert np.array_equal(values[1, 9, 0], values[2, 0, 0])
    for step, stored in enumerate(report["flow_step_metrics"]):
        assert stored["step"] == step
        observed = metrics(values[1, step, 0], expected[step])
        for key, value in observed.items():
            assert abs(stored[key] - value) < 1e-10, (step, key)
    assert report["final_action_metrics"] == report["flow_step_metrics"][-1]
    assert receipt["final_action_metrics"] == report["final_action_metrics"]
    print("verified full SmolVLA output parity and FP32 action-value diagnostic")


if __name__ == "__main__":
    verify()
