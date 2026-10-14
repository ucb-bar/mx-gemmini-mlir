"""Replay SmolVLA step 9 with compiled state and captured FP32 prefix cache.

The full compiled run supplies step-8 state and timestep. Only the prefix
cache changes to the captured FP32 value, isolating that cache's contribution
to final-step drift. This is a diagnostic session, not a model2MLIR capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "smolvla_reference_step_replay_20261014"))
from prepare import prepare as prepare_fp32  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bf16(value: float) -> float:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    word = (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", word))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--compiled-run", type=Path, required=True)
    parser.add_argument("--compiled-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.compiled_run.read_text())
    digests = {(row["program"], row["step"], row["output"]): row
               for row in run["output_digests"]}
    if (run.get("status") != "complete_numeric_qualification_pending"
            or digests[1, 8, 0]["sha256"] != sha(args.compiled_state)):
        raise ValueError("compiled step-8 state is not bound to the full run")
    state = np.fromfile(args.compiled_state, dtype="<f4").reshape(1, 50, 32)
    if not np.isfinite(state).all():
        raise ValueError("compiled state contains nonfinite values")
    time = 1.0
    delta = bf16(-0.1)
    for _ in range(9):
        time = bf16(time + delta)
    time_bytes = struct.pack("<f", time)
    if hashlib.sha256(time_bytes).hexdigest() != digests[1, 8, 2]["sha256"]:
        raise ValueError("compiled step-8 timestep does not match BF16 recurrence")
    receipt = prepare_fp32(args.capture_root, args.output, 9)
    target = args.output / "stages/flow_denoise"
    with np.load(target / "inputs.npz", allow_pickle=False) as archive:
        inputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    inputs["in2"] = np.ascontiguousarray(state)
    inputs["in3"] = np.array([time], dtype="<f4")
    np.savez(target / "inputs.npz", **inputs)
    child_path = target / "session_contract.yaml"
    root_path = args.output / "session_contract.yaml"
    child = yaml.safe_load(child_path.read_text())
    root = yaml.safe_load(root_path.read_text())
    for contract in (child, root):
        contract["provenance"]["diagnostic"] = "compiled_state_fp32_prefix_control"
    child_path.write_text(yaml.safe_dump(child, sort_keys=False))
    root_path.write_text(yaml.safe_dump(root, sort_keys=False))
    receipt.update({
        "schema": "mx_gemmini.smolvla_prefix_factor.v1",
        "diagnostic": "compiled_state_fp32_prefix_control",
        "compiled_run_sha256": sha(args.compiled_run),
        "compiled_state_step8_sha256": sha(args.compiled_state),
        "compiled_timestep_step8_sha256": hashlib.sha256(time_bytes).hexdigest(),
        "derived_inputs_sha256": sha(target / "inputs.npz"),
        "flow_input_sha256": hashlib.sha256(inputs["in2"].tobytes()).hexdigest(),
        "timestep_input": float(inputs["in3"][0]),
        "session_contract_sha256": sha(root_path),
    })
    (args.output / "derived_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
