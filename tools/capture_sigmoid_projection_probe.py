"""Capture a private activation chain spanning an independent projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=19)
    parser.add_argument("--width", type=int, default=37)
    parser.add_argument("--input-padding-rows", type=int, default=0,
                        help="capture a contiguous leading-row slice before the activation and projection")
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0 or args.input_padding_rows < 0 or args.out_dir.exists():
        parser.error("positive extents and a fresh output directory are required")
    import torch
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(args.width, args.width, bias=False)

        def forward(self, x):
            if args.input_padding_rows:
                x = x[:args.rows]
            activated = x * torch.sigmoid(x)
            projected = self.up(x)
            return activated * projected

    torch.set_num_threads(2)
    torch.manual_seed(422)
    out = args.out_dir.resolve()
    write_bundle(Projection().eval(), (torch.randn(args.rows + args.input_padding_rows, args.width),),
                 out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
