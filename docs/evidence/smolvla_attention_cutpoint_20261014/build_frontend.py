"""Capture an isolated masked attention head with model2MLIR.

The operands come from the archived SmolVLA checkpoint cutpoint. Padding and
the additive mask match the MX+VPU source head's physical tile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


class Attention(torch.nn.Module):
    def forward(self, q, kt, mask, v):
        scores = torch.bmm(q, kt)
        scaled = scores * 0.125
        masked = scaled + mask
        probabilities = torch.softmax(masked, dim=-1)
        return torch.bmm(probabilities, v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=int, choices=(113, 163))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model2mlir.resolve()))
    import m2m

    source = HERE / ("cutpoint113.npz" if args.keys == 113 else "cutpoint.npz")
    physical_keys = ((args.keys + 63) // 64) * 64
    with np.load(source, allow_pickle=False) as arrays:
        q = np.zeros((64, 64), np.float32)
        kt = np.zeros((64, physical_keys), np.float32)
        v = np.zeros((physical_keys, 64), np.float32)
        mask = np.full((64, physical_keys), -np.inf, np.float32)
        q[:50] = arrays["q"][0]
        kt[:, :args.keys] = arrays["kt"][0]
        v[:args.keys] = arrays["v"][0]
        mask[:50, :args.keys] = np.where(arrays["mask"][0, 0], 0, -np.inf)
        mask[50:, 0] = 0  # one finite key for discarded padded queries
    inputs = tuple(torch.from_numpy(x.copy())[None] for x in (q, kt, mask, v))
    with torch.no_grad():
        capture = m2m.convert(Attention().eval(), inputs, backend="fx_importer")
    if not capture.ok:
        raise ValueError(f"model2MLIR attention capture failed: {capture.diagnostics}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(capture.mlir_text)
    print(json.dumps({
        "keys": args.keys,
        "cutpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frontend_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "frontend": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
