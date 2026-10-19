"""Capture a private broadcast multiply producer feeding independent projections."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, default=37)
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0 or args.out_dir.exists():
        parser.error("positive extents and a fresh output directory required")
    import torch
    from m2m.capture.bundle import write_bundle

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gain", torch.randn(args.width))
            self.projections = torch.nn.ModuleList(
                torch.nn.Linear(args.width, n, bias=False) for n in (23, 41, 17))

        def forward(self, x, scale):
            hidden = (x * scale) * self.gain
            return torch.cat([projection(hidden) for projection in self.projections], dim=-1)

    torch.manual_seed(421)
    model = Projections().eval()
    x, scale = torch.randn(args.rows, args.width), torch.randn(args.rows, 1)
    out = args.out_dir.resolve()
    write_bundle(model, (x, scale), out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
