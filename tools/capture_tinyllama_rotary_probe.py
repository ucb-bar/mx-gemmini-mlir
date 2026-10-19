"""Capture the real position/frequency operands and an optional independent consumer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.execution_policy import make_policy
from mx_gemmini_support.target_profile import load_profile
from tools.capture_matched_tinyllama_onepass import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-consumer", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory must be fresh")
    import numpy as np
    import torch
    from m2m.capture.bundle import write_bundle

    stage = args.stage.resolve(strict=True)
    receipt = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("inputs.npz", "extra.npz"):
        if sha(stage / name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("captured artifact differs: " + name)
    with np.load(stage / "inputs.npz", allow_pickle=False) as data:
        length = data["in0"].shape[-1]
    with np.load(stage / "extra.npz", allow_pickle=False) as data:
        frequency = data["buf::lm.model.rotary_emb.inv_freq"].copy()
    profile = load_profile(args.profile.resolve(strict=True))
    arguments = (torch.from_numpy(frequency).reshape(1, -1, 1),
                 torch.arange(length, dtype=torch.float32).reshape(1, 1, -1))

    class Outer(torch.nn.Module):
        def forward(self, frequency, position):
            return frequency @ position

    class WithConsumer(torch.nn.Module):
        def forward(self, frequency, position, weight):
            angles = frequency @ position
            return angles, angles @ weight

    if args.include_consumer:
        width = profile["geometry"]["mesh_columns"]
        arguments += (torch.linspace(-0.25, 0.25, length * width).reshape(1, length, width),)
    out = args.out_dir.resolve()
    write_bundle(WithConsumer() if args.include_consumer else Outer(), arguments,
                 out / "capture", capture_regions=False)
    source = (out / "capture/model.mlir").read_text()
    for name, preserve in (("matrix_policy.json", False), ("host_policy.json", True)):
        policy = make_policy(source, profile, preserve_singleton_reductions=preserve)
        (out / name).write_text(json.dumps(policy, indent=2) + "\n")
    (out / "probe_receipt.json").write_text(json.dumps({
        "schema": "mx_gemmini.rotary_probe_capture.v1",
        "source_capture_sha256": sha(stage / "capture_receipt.json"),
        "source_extra_sha256": sha(stage / "extra.npz"),
        "source_inputs_sha256": sha(stage / "inputs.npz"),
        "probe_capture_sha256": sha(out / "capture/capture_receipt.json"),
        "include_consumer": args.include_consumer}, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
