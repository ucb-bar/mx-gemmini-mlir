"""Create a small captured network with a projection between two multiplies."""

import argparse
import json
from pathlib import Path

import torch
from m2m.capture.bundle import write_bundle

from mx_gemmini_support.execution_policy import make_policy
from mx_gemmini_support.target_profile import load_profile


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.up.weight.copy_(torch.arange(1024, dtype=torch.float32).reshape(32, 32)
                                 .remainder(17).sub(8).div(32))

    def forward(self, a, b, x):
        middle = a * b
        up = self.up(x)
        return middle * up


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    values = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    inputs = ((values.remainder(37) - 18) / 31, (values.remainder(43) - 21) / 29,
              (values.remainder(31) - 15) / 23)
    write_bundle(Network().eval(), inputs, args.out / "capture")
    policy = make_policy((args.out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (args.out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")


if __name__ == "__main__":
    main()
