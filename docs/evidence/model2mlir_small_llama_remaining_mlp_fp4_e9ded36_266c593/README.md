# Remaining padded FP4 MLP projections

The [input manifest](inputs/manifest.json) captures all six gate, up, and down
projection operands from a seeded forward pass of model2MLIR `e9ded36`'s
two-layer **small random-weight LLaMA-style model**. This archive executes
the four sites beyond the [first gate/down pair](../model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593/README.md).
Every site is a real rank-two region in the [full captured graph](../model2mlir_small_llama_precision_e9ded36_266c593/small_llama.model2mlir.mlir.gz).

| Site | Region | Logical M×N×K | MX M×N×K | Spike matched BF16 words |
|---|---|---:|---:|---:|
| [Layer-0 up](b0.u/summary.json) | `matmul_7` | 8×344×128 | 32×352×128 | 11,264/11,264 |
| [Layer-1 gate](b1.g/summary.json) | `matmul_15` | 8×344×128 | 32×352×128 | 11,264/11,264 |
| [Layer-1 up](b1.u/summary.json) | `matmul_16` | 8×344×128 | 32×352×128 | 11,264/11,264 |
| [Layer-1 down](b1.dn/summary.json) | `matmul_17` | 8×128×344 | 32×128×352 | 4,096/4,096 |

With the first pair, **all six shape-padding candidates** in the current
small LLaMA precision proposal compile as individual FP4 objects and execute
on the reference implementation’s pinned MX Spike model. Across the six, all **53,248 padded BF16
words** matched the independent Radiance MX arithmetic reference; 13,056 are
logical model outputs. Packed resources, goldens, bound MLIR, objects, ELFs,
Spike logs, and result manifests are archived for the four sites here.

Replay the [receipt](receipt.json) with:

```sh
python -m tools.qualify_small_llama_mlp_fp4_padding \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQ_ROOT" \
  --reference-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt "$MX_OPT" --out-dir "$OUT_DIR" \
  --remaining-mlp
```

These are six separate projection programs. The nonlinear MLP path, attention,
other model operations, and full inference session remain uncompiled. No
TinyLlama checkpoint or FPGA was used.
