"""Capture a batched matmul with source-explicit grouped RHS broadcasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for axis in ("groups", "repeat", "m", "n", "k"):
        parser.add_argument("--" + axis, required=True, type=int)
    parser.add_argument("--transpose-rhs", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists() or min(args.groups, args.repeat, args.m, args.n, args.k) <= 0:
        parser.error("fresh output directory and positive static shapes required")
    import torch
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Grouped(torch.nn.Module):
        def forward(self, lhs, rhs):
            rows, cols = (args.n, args.k) if args.transpose_rhs else (args.k, args.n)
            rhs = rhs.unsqueeze(1).expand(args.groups, args.repeat, rows, cols)
            rhs = rhs.reshape(args.groups * args.repeat, rows, cols)
            if args.transpose_rhs:
                rhs = rhs.transpose(-1, -2)
            return torch.bmm(lhs, rhs)

    torch.manual_seed(3182)
    lhs = torch.randn(args.groups * args.repeat, args.m, args.k)
    rhs = torch.randn(args.groups, *((args.n, args.k) if args.transpose_rhs else (args.k, args.n)))
    out = args.out_dir.resolve()
    write_bundle(Grouped().eval(), (lhs, rhs), out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    policy["matrix_formats"] = {region: "mxfp8" for region in policy["matrix_formats"]}
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"out": str(out), "shape_bmnk": [args.groups * args.repeat, args.m, args.n, args.k],
                      "repeat": args.repeat, "transpose": args.transpose_rhs}))


if __name__ == "__main__":
    main()
