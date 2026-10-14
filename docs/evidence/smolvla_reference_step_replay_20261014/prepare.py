"""Derive one flow-step Spike replay from a captured SmolVLA FP32 trajectory.

This is a diagnostic input substitution, not a new model2MLIR capture. The
selected FP32 output of step N-1 is the carried flow-state input of step N;
the prefix cache is unchanged and the timestep follows the loader's Euler
schedule. Large immutable capture files are hard-linked within the task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(capture_root: Path, output: Path, step: int) -> dict:
    capture_root = capture_root.resolve(strict=True)
    source = capture_root / "stages/flow_denoise"
    original = yaml.safe_load((source / "session_contract.yaml").read_text())
    root = yaml.safe_load((capture_root / "session_contract.yaml").read_text())
    if (original["version"] != 1 or root["version"] != 2
            or original["steps"] != 10 or root["stages"] != [
                "prefix_encode", "flow_denoise", "action_decode"]
            or [(s["name"], s["output_index"]) for s in original["states"]] != [
                ("prefix_kv_cache", 1), ("flow_state", 0), ("timestep", 2)]
            or original["quality"]["key"] != "actions"
            or original["quality"]["output_index"] != 0
            or original["streams"]):
        raise ValueError("source is not the selected ten-step SmolVLA flow capture")
    if not 0 <= step < 10:
        raise ValueError("reference step must be in 0..9")
    with np.load(source / "session_quality_fp32.npz", allow_pickle=False) as data:
        reference = np.ascontiguousarray(data["actions"])
    with np.load(source / "session_goldens.npz", allow_pickle=False) as data:
        correctness = np.ascontiguousarray(data["actions"])
    if reference.shape != (10, 1, 50, 32) or reference.dtype != np.float32:
        raise ValueError("FP32 reference has the wrong shape or dtype")
    if not np.array_equal(reference, correctness):
        raise ValueError("source correctness and FP32 quality trajectories differ")
    with np.load(source / "inputs.npz", allow_pickle=False) as data:
        inputs = {f"in{i}": np.ascontiguousarray(data[f"in{i}"]) for i in range(4)}
    if (inputs["in0"].shape != (1, 113)
            or inputs["in1"].shape != (2, 16, 1, 113, 5, 64)
            or inputs["in2"].shape != (1, 50, 32)
            or inputs["in3"].shape != (1,)
            or inputs["in3"].dtype != np.float32
            or float(inputs["in3"][0]) != 1.0):
        raise ValueError("source flow initial inputs differ from the pinned ABI")
    if step:
        inputs["in2"] = np.ascontiguousarray(reference[step - 1])
    for _ in range(step):
        inputs["in3"] = inputs["in3"] + np.float32(-0.1)

    output.mkdir(parents=True, exist_ok=False)
    target = output / "stages/flow_denoise"
    target.mkdir(parents=True)
    for name in ("model.mlir", "weights.safetensors",
                 "weights.safetensors.manifest.json", "extra.npz",
                 "input_order.json", "meta.json"):
        os.link(source / name, target / name)
    expected = np.ascontiguousarray(reference[step:step + 1])
    np.savez(target / "inputs.npz", **inputs)
    np.save(target / "golden.npy", expected[0])
    np.savez(target / "session_inputs.npz")
    np.savez(target / "session_goldens.npz", actions=expected)
    np.savez(target / "session_quality_fp32.npz", actions=expected)
    child = dict(original)
    child["steps"] = 1
    child["stage_schedule"] = [{"name": "flow_denoise", "steps": 1,
                                "execution": "compiled_recurrent", "timed": True}]
    child["parameters"] = {"denoise_steps": 10, "batch": 1,
                           "derived_reference_step": step}
    digest = hashlib.sha256(expected.tobytes()).hexdigest()
    child["correctness"]["reference_sha256"] = digest
    child["quality"]["reference_sha256"] = digest
    child["paper_ready"] = False
    child["provenance"] = {**child["provenance"],
                           "diagnostic": "isolated_fp32_state_input",
                           "derived_reference_step": step}
    (target / "session_contract.yaml").write_text(
        yaml.safe_dump(child, sort_keys=False))
    derived_root = {"version": 2, "kind": "action_chunk",
                    "paper_ready": False, "stages": ["flow_denoise"],
                    "stage_schedule": child["stage_schedule"],
                    "programs": [{"name": "flow_denoise",
                                  "bundle": "stages/flow_denoise", "steps": 1}],
                    "bindings": [], "states": [
                        {"name": state["name"]} for state in child["states"]],
                    "streams": [], "quality": {"scope": "trajectory",
                                                  "program": "flow_denoise"},
                    "parameters": child["parameters"],
                    "provenance": child["provenance"]}
    (output / "session_contract.yaml").write_text(
        yaml.safe_dump(derived_root, sort_keys=False))
    receipt = {"schema": "mx_gemmini.smolvla_reference_step_replay.v1",
               "source_session_contract_sha256": _sha(capture_root / "session_contract.yaml"),
               "source_child_contract_sha256": _sha(source / "session_contract.yaml"),
               "source_quality_sha256": _sha(source / "session_quality_fp32.npz"),
               "source_inputs_sha256": _sha(source / "inputs.npz"),
               "model_mlir_sha256": _sha(source / "model.mlir"),
               "reference_step": step, "reference_output_sha256": digest,
               "derived_inputs_sha256": _sha(target / "inputs.npz"),
               "initial_cache_sha256": hashlib.sha256(inputs["in1"].tobytes()).hexdigest(),
               "initial_cache_bf16_sha256": hashlib.sha256((
                   (inputs["in1"].view(np.uint32) + np.uint32(0x7FFF)
                    + ((inputs["in1"].view(np.uint32) >> 16) & np.uint32(1)))
                   >> np.uint32(16)).astype(np.uint16).tobytes()).hexdigest(),
               "flow_input_sha256": hashlib.sha256(inputs["in2"].tobytes()).hexdigest(),
               "timestep_input": float(inputs["in3"][0]),
               "session_contract_sha256": _sha(output / "session_contract.yaml")}
    (output / "derived_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.capture_root, args.output, args.step), sort_keys=True))


if __name__ == "__main__":
    main()
