# Padded FP4 MLP projections from current model2MLIR

The [input manifest](inputs/manifest.json) records layer-0 MLP gate and down
projection operands extracted from a seeded forward pass of model2MLIR
`e9ded36`'s **small random-weight LLaMA-style model**. Both `linalg.matmul`
regions occur in the [captured full graph](../model2mlir_small_llama_precision_e9ded36_266c593/small_llama.model2mlir.mlir.gz).
These are complete individual projections, not a TinyLlama checkpoint or a
whole-model executable.

| Site | Model2MLIR region | Logical M×N×K | MX M×N×K | FP4 output check |
|---|---|---:|---:|---:|
| [gate](g/summary.json) | `matmul_6` | 8×344×128 | 32×352×128 | 11,264/11,264 BF16 words |
| [down](dn/summary.json) | `matmul_8` | 8×128×344 | 32×128×352 | 4,096/4,096 BF16 words |

The compiler zero-pads M to the FP4 mesh tile and N/K to 32-element MX
groups, quantizes the padded operands with pinned MXQuant, captures the
isolated padded matmul using the current model2MLIR frontend, and lowers its
typed MX contract to a data-free RV64 object. Nicolas's pinned MX Spike
extension ran each standalone ELF. Every padded BF16 output matched the
independent Radiance MX arithmetic reference; output rows and columns outside
the logical model dimensions are zero. Logical PyTorch f32 versus MX BF16
errors are recorded in each [summary](g/summary.json) and
[summary](dn/summary.json).

The archived bundles include packed operand and scale resources, independent
goldens, bound MLIR, objects, ELF files, Spike logs, and receipts. The
[replay tool](../../../tools/qualify_small_llama_mlp_fp4_padding.py) recaptures
the model operands, recompiles both sites, runs Spike, and compares generated
objects, ELFs, goldens, and summaries with this archive:

```sh
python -m tools.qualify_small_llama_mlp_fp4_padding \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQ_ROOT" \
  --reference-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt "$MX_OPT" --out-dir "$OUT_DIR"
```

The model's SiLU, gate multiply, residual path, other projections, attention,
and recurrent state are outside these two programs. No FPGA was used.
