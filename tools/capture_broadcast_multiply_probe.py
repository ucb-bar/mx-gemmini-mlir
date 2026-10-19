"""Capture independent row/channel multiply chains under a BF16 call policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def broadcast_source(*, rows=3, width=137, middle_index=1, first_swapped=False):
    """Independent row/channel broadcasts with either source operand order."""
    full, row, channel = f"tensor<1x{rows}x{width}xf32>", f"tensor<1x{rows}x1xf32>", f"tensor<{width}xf32>"
    identity = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>"
    rowmap = "affine_map<(d0, d1, d2) -> (d0, d1, 0)>"
    colmap = "affine_map<(d0, d1, d2) -> (d2)>"
    first = [("%a", full, identity), ("%b", row, rowmap)]
    second = [("%m", full, identity), ("%c", channel, colmap)]
    if first_swapped:
        first.reverse()
    if middle_index == 1:
        second.reverse()
    blocks = []
    for i, (name, inputs) in enumerate(zip(("m", "o"), (first, second), strict=True)):
        blocks.append(f'''
          %e{i} = "tensor.empty"() : () -> {full}
          %{name} = "linalg.generic"({inputs[0][0]}, {inputs[1][0]}, %e{i})
            <{{indexing_maps = [{inputs[0][2]}, {inputs[1][2]}, {identity}],
              iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                                #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}}> ({{
            ^bb0(%x: f32, %y: f32, %init: f32):
              %v = "arith.mulf"(%x, %y) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32
              "linalg.yield"(%v) : (f32) -> ()
          }}) {{prov.region_id = "mul_{i}", prov.op = "mul", prov.aten = "aten.mul.Tensor"}} :
            ({inputs[0][1]}, {inputs[1][1]}, {full}) -> {full}
        ''')
    return f'''builtin.module {{
      func.func @forward(%a: {full}, %b: {row}, %c: {channel}) -> {full} {{
        {''.join(blocks)}
        func.return %o : {full}
      }}
    }}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--middle-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--first-swapped", action="store_true")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.rows <= 0 or args.width <= 0:
        parser.error("capture needs positive extents and a fresh output directory")
    import numpy as np
    import torch

    rng = np.random.default_rng(17)
    x = rng.normal(size=(1, args.rows, args.width)).astype(np.float32)
    scale = rng.uniform(0.125, 2.0, size=(1, args.rows, 1)).astype(np.float32)
    weight = rng.normal(size=(args.width,)).astype(np.float32)
    # Include signed zeros, BF16 subnormals and narrowing tie cases. The
    # reference executes each original BF16 multiplication boundary separately.
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x00010000, 0x80010000,
                      0x3f808000, 0x3f818000, 0xbf808000], dtype=np.uint32)
    x.reshape(-1)[:min(x.size, words.size)] = words[:min(x.size, words.size)].view(np.float32)
    if args.fallback:
        weight[-1] = np.float32(2.0 ** 62)
    tx, ts, tw = (torch.from_numpy(v).to(torch.bfloat16).float() for v in (x, scale, weight))
    first = (tx * ts).to(torch.bfloat16).float()
    golden = (first * tw).to(torch.bfloat16).float().numpy()
    inputs = [x, scale, weight]
    if args.first_swapped:
        inputs[:2] = reversed(inputs[:2])
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    (out / "source.mlir").write_text(broadcast_source(
        rows=args.rows, width=args.width, middle_index=args.middle_index, first_swapped=args.first_swapped))
    np.savez(out / "inputs.npz", **{f"in{i}": v for i, v in enumerate(inputs)})
    np.save(out / "bf16_policy_golden.npy", golden)
    receipt = {"schema": "mx_gemmini.broadcast_multiply_capture.v1", "seed": 17,
               "reference": "two separate BF16-rounded calls using PyTorch", "torch_version": torch.__version__,
               "shape": list(x.shape), "input_shapes": [list(v.shape) for v in inputs],
               "middle_index": args.middle_index, "first_swapped": args.first_swapped,
               "fallback": args.fallback,
               "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "files_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in
                                ("source.mlir", "inputs.npz", "bf16_policy_golden.npy")}}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()

