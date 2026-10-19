"""Capture a matrix result crossing private views into a residual addition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=19)
    parser.add_argument("--bias-broadcast", action="store_true")
    parser.add_argument("--shared-projections", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.out_dir.exists():
        parser.error("positive rows and fresh output required")
    import torch
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Section(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(37, 23, bias=False)
            if args.shared_projections:
                self.second = torch.nn.Linear(37, 23, bias=False)

        def forward(self, x, residual):
            result = self.linear(x).reshape(1, args.rows, 23)
            if args.shared_projections:
                other = self.second(x).reshape(1, args.rows, 23)
                return torch.cat((residual + result, residual + other), dim=-1)
            return result + residual if args.bias_broadcast else residual + result

    torch.manual_seed(812)
    model = Section().eval()
    inputs = (torch.randn(args.rows, 37),
              torch.randn(23) if args.bias_broadcast else torch.randn(1, args.rows, 23))
    out = args.out_dir.resolve()
    write_bundle(model, inputs, out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
