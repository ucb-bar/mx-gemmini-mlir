"""Verify the archived isolated SmolVLA flow-step replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("flow outputs must have matching finite shapes")
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    difference = lhs - rhs
    norm = max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = max(float(np.sqrt(np.mean(rhs * rhs))), 1e-300)
    return {"mae": float(np.mean(np.abs(difference))), "rmse": rmse,
            "normalized_rmse": rmse / reference_rms,
            "max_abs": float(np.max(np.abs(difference))),
            "cosine": float(np.clip(np.sum(lhs * rhs) / norm, -1.0, 1.0))}


def _bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def timestep_diagnostic(full_run: dict) -> dict:
    """Check the f32 ABI output against ten BF16-rounded VPU additions."""
    digests = {(row["program"], row["step"], row["output"]): row["sha256"]
               for row in full_run["output_digests"]}
    delta = _bf16(-0.1)
    bf16_time = 1.0
    f32_time = np.float32(1.0)
    rows = []
    for step in range(10):
        bf16_time = _bf16(bf16_time + delta)
        f32_time = np.float32(f32_time + np.float32(-0.1))
        actual = digests[1, step, 2]
        predicted = hashlib.sha256(struct.pack("<f", bf16_time)).hexdigest()
        if actual != predicted:
            raise ValueError(f"flow timestep {step} does not match BF16 VPU addition")
        rows.append({"step": step, "compiled_bf16": bf16_time,
                     "reference_f32": float(f32_time)})
    return {"matched_steps": len(rows), "final_compiled": rows[-1]["compiled_bf16"],
            "final_reference": rows[-1]["reference_f32"],
            "final_abs_difference": abs(rows[-1]["compiled_bf16"] - rows[-1]["reference_f32"])}


def verify() -> dict:
    receipt = json.loads((HERE / "receipt.json").read_text())
    if receipt.get("schema") != "mx_gemmini.smolvla_reference_step_replay_result.v1":
        raise ValueError("unexpected replay receipt schema")
    for name, digest in receipt["artifact_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"archived artifact changed: {name}")
    run = json.loads((HERE / "run.json").read_text())
    link = json.loads((HERE / "link.json").read_text())
    source_run = json.loads((HERE / "full_run.json").read_text())
    source_link = json.loads((HERE / "full_link.json").read_text())
    derived = json.loads((HERE / "derived_replay_receipt.json").read_text())
    if (run["status"] != "complete_numeric_qualification_pending"
            or run["link_receipt_sha256"] != sha(HERE / "link.json")
            or source_run["link_receipt_sha256"] != sha(HERE / "full_link.json")
            or [(row["name"], row["steps"]) for row in link["programs"]]
            != [("flow_denoise", 1)]
            or derived["reference_step"] != 9
            or link["programs"][0]["capture_model_sha256"] != derived["model_mlir_sha256"]
            or link["stage_object_sha256"]["flow_denoise"]
            != source_link["stage_object_sha256"]["flow_denoise"]
            or link["profile_sha256"] != source_link["profile_sha256"]
            or link["shared_weight_sha256"] != source_link["shared_weight_sha256"]):
        raise ValueError("isolated and full flow runs have inconsistent identities")
    selected = [row for row in run["output_dumps"]
                if (row["program"], row["step"], row["output"]) == (0, 0, 0)]
    if len(selected) != 1 or selected[0]["bytes"] != 6400:
        raise ValueError("isolated flow output dump is missing")
    isolated = HERE / "isolated_step9.bin"
    full = HERE / "full_step9.bin"
    expected = HERE / "fp32_step9.bin"
    if (isolated.stat().st_size != 6400 or full.stat().st_size != 6400
            or expected.stat().st_size != 6400
            or selected[0]["sha256"] != sha(isolated)
            or derived["reference_output_sha256"] != sha(expected)):
        raise ValueError("flow output or FP32 reference bytes differ")
    source_digest = [row for row in source_run["output_digests"]
                     if (row["program"], row["step"], row["output"]) == (1, 9, 0)]
    if len(source_digest) != 1 or source_digest[0]["sha256"] != sha(full):
        raise ValueError("full-session step-nine output differs from its run receipt")
    compiled_cache = [row for row in source_run["output_digests"]
                      if (row["program"], row["step"], row["output"]) == (1, 0, 1)]
    if (len(compiled_cache) != 1 or
            compiled_cache[0]["sha256"] == derived["initial_cache_bf16_sha256"]):
        raise ValueError("isolated replay did not replace the compiled prefix cache")
    if timestep_diagnostic(source_run) != receipt["timestep_diagnostic"]:
        raise ValueError("archived VPU timestep diagnostic differs from run hashes")
    arrays = [np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 50, 32)
              for path in (isolated, full, expected)]
    measured_metrics = {"isolated_vs_fp32": metrics(arrays[0], arrays[2]),
                        "full_vs_fp32": metrics(arrays[1], arrays[2])}
    for key, values in measured_metrics.items():
        for metric, measured in values.items():
            if abs(measured - receipt["metrics"][key][metric]) > 1e-12:
                raise ValueError(f"archived {key}.{metric} differs from data")
    return measured_metrics


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
