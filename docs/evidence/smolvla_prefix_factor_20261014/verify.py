"""Verify the SmolVLA compiled-state / FP32-prefix control archive."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "smolvla_full_action_values_20261014"
REFERENCE = HERE.parent / "smolvla_reference_step_replay_20261014"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()):
        raise ValueError("factorial outputs must have matching finite shapes")
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    delta = lhs - rhs
    return {"cosine": float(np.dot(lhs, rhs) /
                            (np.linalg.norm(lhs) * np.linalg.norm(rhs))),
            "normalized_rmse": float(np.sqrt(np.mean(delta * delta)) /
                                     np.sqrt(np.mean(rhs * rhs)))}


def verify() -> dict:
    receipt = json.loads((HERE / "receipt.json").read_text())
    if (receipt.get("schema") != "mx_gemmini.smolvla_prefix_factor_result.v1"
            or receipt.get("cycle_ratio_valid") is not False
            or receipt.get("numerical_qualification") != "not_established"):
        raise ValueError("unexpected factorial control receipt")
    for name, digest in receipt["artifact_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"factorial artifact changed: {name}")
    run = json.loads((HERE / "run.json").read_text())
    link = json.loads((HERE / "link.json").read_text())
    source_run = json.loads((FULL / "tap_run.json").read_text())
    source_link = json.loads((FULL / "tap_link.json").read_text())
    derived = json.loads((HERE / "derived_replay_receipt.json").read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run["link_receipt_sha256"] != sha(HERE / "link.json")
            or run["timing_config"] != source_run["timing_config"]
            or run["timing_overrides"] != source_run["timing_overrides"]
            or link["profile_sha256"] != source_link["profile_sha256"]
            or link["shared_weight_sha256"] != source_link["shared_weight_sha256"]
            or link["stage_object_sha256"]["flow_denoise"]
            != source_link["stage_object_sha256"]["flow_denoise"]
            or derived["schema"] != "mx_gemmini.smolvla_prefix_factor.v1"
            or derived["compiled_run_sha256"] != sha(FULL / "tap_run.json")
            or derived["reference_step"] != 9):
        raise ValueError("factorial input, object, or Spike identities differ")
    digests = {(row["program"], row["step"], row["output"]): row["sha256"]
               for row in source_run["output_digests"]}
    if (derived["compiled_state_step8_sha256"] !=
            sha(FULL / "output_dumps/p1_s8_o0.bin")
            or derived["compiled_state_step8_sha256"] != digests[1, 8, 0]
            or derived["compiled_timestep_step8_sha256"] != digests[1, 8, 2]
            or derived["initial_cache_bf16_sha256"] == digests[0, 0, 1]):
        raise ValueError("factorial control did not preserve compiled state or reset cache")
    output_dumps = {(row["program"], row["step"], row["output"]): row
                    for row in run["output_dumps"]}
    for output, name, size in ((0, "flow_step9.bin", 6400),
                               (2, "timestep_step9.bin", 4)):
        row = output_dumps[0, 0, output]
        if row["bytes"] != size or row["sha256"] != sha(HERE / name):
            raise ValueError("factorial output bytes differ from Spike")
    if sha(HERE / "timestep_step9.bin") != digests[1, 9, 2]:
        raise ValueError("factorial timestep output differs from the full run")
    if (hashlib.sha256(gzip.decompress(
            (HERE / "spike.log.gz").read_bytes())).hexdigest()
            != run["log_sha256"]
            or sha(HERE / "gemmini_perf.txt") != run["timing_report_sha256"]):
        raise ValueError("factorial Spike log or timing report changed")
    prior_time = derived["timestep_input"]
    if not np.isfinite(prior_time):
        raise ValueError("factorial timestep input is nonfinite")
    expected = np.fromfile(REFERENCE / "fp32_step9.bin", dtype="<f4")
    factor = np.fromfile(HERE / "flow_step9.bin", dtype="<f4")
    full = np.fromfile(FULL / "output_dumps/p1_s9_o0.bin", dtype="<f4")
    if not (len(expected) == len(factor) == len(full) == 1600):
        raise ValueError("factorial flow output shape differs")
    observed = {"factor_vs_fp32": metrics(factor, expected),
                "full_vs_fp32": metrics(full, expected),
                "factor_vs_full": metrics(factor, full)}
    for name, values in observed.items():
        for metric, value in values.items():
            if abs(value - receipt["metrics"][name][metric]) > 1e-12:
                raise ValueError(f"factorial metric differs: {name}.{metric}")
    return {**observed,
            "isolated_spike_cycles": run["total_performance"]["cycles"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
