"""Verify archived 33-step TinyLlama logits and FP32 diagnostics."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014"
sys.path.insert(0, str(ROOT))
from tools.compare_tinyllama_session import metrics  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.tinyllama_full_logits.v1"
    assert receipt["numerical_qualification"] == "not_established"
    assert receipt["scope"] == "one prefill and 32 decode steps"
    assert receipt["output_digests_match_fast_compiler"] is True
    for name, expected in receipt["artifact_sha256"].items():
        assert sha(HERE / name) == expected, name

    tap = json.loads((HERE / "tap_run.json").read_text())
    link = json.loads((HERE / "tap_link.json").read_text())
    fast_path = FAST / "compiler/mx_session_run.json"
    fast_link_path = FAST / "compiler_link.json"
    fast = json.loads(fast_path.read_text())
    fast_link = json.loads(fast_link_path.read_text())
    assert sha(fast_path) == receipt["fast_run_sha256"]
    assert sha(fast_link_path) == receipt["fast_link_sha256"]
    assert tap["link_receipt_sha256"] == sha(HERE / "tap_link.json")
    assert tap["status"] == fast["status"] == "complete_numeric_qualification_pending"
    assert tap["complete"] is fast["complete"] is True
    assert tap["exit_code"] == fast["exit_code"] == 0
    assert tap["expected_events"] == tap["observed_events"] == 99
    assert tap["timing_mode"] == fast["timing_mode"] == "both"
    assert tap["timing_config"] == fast["timing_config"] == "e4m3_vpu"
    assert tap["timing_overrides"] == fast["timing_overrides"] == "mem.host_tracking=0"
    assert tap["output_digests"] == fast["output_digests"]
    assert link["stage_object_sha256"] == fast_link["stage_object_sha256"]
    assert link["shared_weight_sha256"] == fast_link["shared_weight_sha256"]
    assert link["dump_limit_bytes"] == 128000
    assert tap["total_performance"]["cycles"] == receipt["tap_cycles"]
    assert fast["total_performance"]["cycles"] == receipt["fast_compiler_cycles"]
    log = gzip.decompress((HERE / "tap_spike.log.gz").read_bytes())
    assert hashlib.sha256(log).hexdigest() == tap["log_sha256"]
    assert b"DONE\n" in log

    report = json.loads((HERE / "fp32_comparison.json").read_text())
    assert report["schema"] == "mx_gemmini.tinyllama_session_fp32_comparison.v1"
    assert report["run_receipt_sha256"] == sha(HERE / "tap_run.json")
    assert report["link_receipt_sha256"] == sha(HERE / "tap_link.json")
    assert report["fast_run_receipt_sha256"] == receipt["fast_run_sha256"]
    assert report["fast_link_receipt_sha256"] == receipt["fast_link_sha256"]
    assert report["prefill_reference_sha256"] == sha(HERE / "prefill_reference.npy")
    assert report["decode_reference_sha256"] == sha(HERE / "decode_reference.npz")
    assert report["output_digests_match_fast_compiler"] is True
    prefill = np.load(HERE / "prefill_reference.npy", allow_pickle=False)
    with np.load(HERE / "decode_reference.npz", allow_pickle=False) as archive:
        decode = archive["logits"]
    assert prefill.shape == (1, 32000) and prefill.dtype == np.float32
    assert decode.shape == (32, 1, 32000) and decode.dtype == np.float32
    selected = {(row["program"], row["step"], row["output"]): row
                for row in tap["output_dumps"]}
    assert len(tap["output_dumps"]) == len(selected) == 33
    assert set(selected) == {(0, 0, 0)} | {(1, step, 0) for step in range(32)}
    values = {}
    for key, row in selected.items():
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = HERE / relative
        assert row["bytes"] == path.stat().st_size == 128000
        assert sha(path) == row["sha256"]
        values[key] = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 32000).copy()
    for stored, observed in [(report["prefill_metrics"],
                              metrics(values[0, 0, 0], prefill))]:
        for key, value in observed.items():
            assert abs(stored[key] - value) < 1e-10, key
    assert receipt["prefill_metrics"] == report["prefill_metrics"]
    assert len(report["decode_step_metrics"]) == 32
    for step, stored in enumerate(report["decode_step_metrics"]):
        assert stored["step"] == step
        observed = metrics(values[1, step, 0], decode[step])
        for key, value in observed.items():
            assert abs(stored[key] - value) < 1e-10, (step, key)
    cosines = [row["cosine"] for row in report["decode_step_metrics"]]
    summary = report["decode_summary"]
    assert summary["steps"] == 32
    assert abs(summary["cosine_min"] - min(cosines)) < 1e-10
    assert abs(summary["cosine_mean"] - float(np.mean(cosines))) < 1e-10
    mean_nrmse = float(np.mean(
        [row["normalized_rmse"] for row in report["decode_step_metrics"]]))
    assert abs(summary["normalized_rmse_mean"] - mean_nrmse) < 1e-10
    assert summary["top_token_matches"] == sum(
        row["top_token_equal"] for row in report["decode_step_metrics"])
    assert receipt["decode_summary"] == summary
    print("verified full TinyLlama logit values and FP32 diagnostics")


if __name__ == "__main__":
    verify()
