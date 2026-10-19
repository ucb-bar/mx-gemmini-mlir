"""Capture independent projections of one live input, including matrix tails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, default=37)
    parser.add_argument("--bf16-projections", action="store_true",
                        help="capture BF16 linear operations with a shared narrowed input and F32 output")
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0 or args.out_dir.exists():
        parser.error("positive extents and a fresh output directory are required")
    import torch
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = torch.nn.ModuleList(
                torch.nn.Linear(args.width, n, bias=False,
                                dtype=torch.bfloat16 if args.bf16_projections else torch.float32)
                for n in (23, 41, 17))

        def forward(self, x):
            if args.bf16_projections:
                x = x.to(torch.bfloat16)
            a = self.projections[0](x)
            b = self.projections[1](x)
            c = self.projections[2](x)
            return torch.cat((a, b, c), dim=-1).float()

    torch.manual_seed(421)
    model = Projections().eval()
    x = torch.randn(args.rows, args.width)
    out = args.out_dir.resolve()
    write_bundle(model, (x,), out / "capture", capture_regions=False)
    profile = load_profile(args.profile.resolve(strict=True))
    policy = make_policy((out / "capture/model.mlir").read_text(), profile)
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
