"""Capture a source sigmoid/multiply chain for transport-only qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=137)
    parser.add_argument("--exhaustive-bf16-gate", action="store_true")
    parser.add_argument("--section-inputs", type=Path)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or args.elements <= 0:
        parser.error("output must be fresh and element count positive")
    import m2m
    import numpy as np
    import torch

    torch.set_num_threads(2)
    source_sha = None
    if args.section_inputs:
        with np.load(args.section_inputs, allow_pickle=False) as arrays:
            x, up = arrays["in0"].copy(), arrays["in2"].copy()
        source_sha = hashlib.sha256(args.section_inputs.read_bytes()).hexdigest()
        scope = "saved_fp32_section_inputs_not_quantized_intermediates_or_full_model"
    elif args.exhaustive_bf16_gate:
        positive = np.arange(0x4181, dtype=np.uint32) << 16
        words = np.concatenate((positive, positive | np.uint32(0x80000000)))
        x = words.view(np.float32).reshape(1, -1)
        choices = np.array([0., -0., 1., -1., 2.**-100, 2.**60, -(2.**60), 0.3], np.float32)
        up = np.resize(choices, x.size).reshape(x.shape)
        scope = "all33538_bf16_gate_encodings_with_abs_at_most16_and_varied_finite_second_operand"
    else:
        rng = np.random.default_rng(420)
        x = rng.uniform(-12, 12, (1, args.elements)).astype(np.float32)
        up = rng.normal(size=x.shape).astype(np.float32)
        scope = "independent_finite_pointwise_fixture"
    if x.shape != up.shape or x.dtype != np.float32 or up.dtype != np.float32:
        raise ValueError("section needs equally shaped FP32 gate and up operands")
    if args.fallback:
        x = x.copy()
        x.flat[-1] = 20.0
        scope += "_late_out_of_guard_fallback"

    class Chain(torch.nn.Module):
        def forward(self, gate, other):
            return (gate * torch.sigmoid(gate)) * other

    result = m2m.convert(Chain().eval(), (torch.from_numpy(x), torch.from_numpy(up)), backend="fx_importer")
    if not result.ok or not result.mlir_text:
        raise ValueError("pointwise source did not convert to complete typed MLIR")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "source.mlir").write_text(result.mlir_text)
    np.savez(args.out_dir / "inputs.npz", in0=x, in1=up)
    (args.out_dir / "receipt.json").write_text(json.dumps({
        "scope": scope, "section_inputs_sha256": source_sha, "shape": list(x.shape),
        "all_inputs_finite": bool(np.isfinite(x).all() and np.isfinite(up).all()),
        "maximum_gate_magnitude": float(np.max(np.abs(x))),
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in args.out_dir.iterdir() if p.is_file()},
    }, indent=2) + "\n")
    print(str(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
