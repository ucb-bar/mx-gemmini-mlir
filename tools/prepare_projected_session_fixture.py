"""Capture a recurrent matrix fixture whose dead weight shifts all state slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("fixture output must be fresh")
    import torch
    import yaml
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(32))
                self.linear.bias.fill_(0.125)

        def forward(self, x, state):
            result = self.linear(x) + state
            return result, result + 0.5

    model = Recurrent().eval()
    out = args.out_dir.resolve()
    profile = load_profile(args.profile.resolve(strict=True))
    coordinates = torch.arange(16 * 32).reshape(16, 32)
    pattern = (coordinates % 4).to(torch.float32) * 0.0625
    prefill_x = 0.25 + pattern
    initial = (coordinates % 7).to(torch.float32) * 0.0625
    state = initial.clone()
    with torch.no_grad():
        prefill = model(prefill_x, state)
        state = prefill[1]
        frames = [value + torch.roll(pattern, shifts=index, dims=1)
                  for index, value in enumerate((0.25, 0.5, 0.75))]
        trajectory = []
        for frame in frames:
            logits, state = model(frame, state)
            trajectory.append(logits.numpy())
    for name, inputs in (("prefill", (prefill_x, initial)),
                         ("decode", (frames[0], torch.full_like(prefill_x, -7)))):
        stage = out / "stages" / name
        write_bundle(model, inputs, stage, capture_regions=False)
        manifest = json.loads((stage / "weights.safetensors.manifest.json").read_text())
        input_args = [int(index) for index, row in manifest.items() if row["kind"] == "input"]
        if len(input_args) != 2:
            raise ValueError("fixture does not expose the intended stream/state inputs")
        if name == "decode":
            np.savez(stage / "session_inputs.npz", frames=np.stack([frame.numpy() for frame in frames]))
            np.savez(stage / "session_goldens.npz", logits=np.stack(trajectory))
            child = {
                "version": 1, "kind": "autoregressive_decode", "paper_ready": False,
                "stages": [name], "steps": len(frames), "inputs": "session_inputs.npz",
                "states": [{"name": "state", "input_arg": input_args[1], "output_index": 1}],
                "streams": [{"name": "x", "input_arg": input_args[0], "key": "frames"}],
                "quality": {"scope": "trajectory", "golden": "session_goldens.npz",
                            "key": "logits", "output_index": 0},
            }
            (stage / "session_contract.yaml").write_text(yaml.safe_dump(child))
            decode_state_arg = input_args[1]
        policy = make_policy((stage / "model.mlir").read_text(), profile)
        (out / (name + "_policy.json")).write_text(json.dumps(policy, indent=2) + "\n")
    root = {
        "version": 2, "kind": "autoregressive_decode", "paper_ready": False,
        "stages": ["prefill", "decode"],
        "stage_schedule": [{"name": name, "steps": steps, "execution": "compiled", "timed": True}
                           for name, steps in (("prefill", 1), ("decode", len(frames)))],
        "programs": [{"name": name, "bundle": "stages/" + name, "steps": steps}
                     for name, steps in (("prefill", 1), ("decode", len(frames)))],
        "bindings": [{"name": "state", "from": {"program": "prefill", "output_index": 1},
                      "to": {"program": "decode", "input_arg": decode_state_arg}}],
        "states": [{"name": "state"}], "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }
    (out / "session_contract.yaml").write_text(yaml.safe_dump(root))
    np.savez(out / "expected.npz", prefill=prefill[0].numpy(), decode=np.stack(trajectory))
    print(json.dumps({"out": str(out), "steps": [1, len(frames)],
                      "initial_decode_state_sentinel": -7, "bound_state_arg": decode_state_arg}))


if __name__ == "__main__":
    main()
