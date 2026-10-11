# Nicolas generic FP6 debug data through current model2MLIR

Nicolas's pinned `matmul_ws_mx_generic.c` uses a fixed MMIO Gemmini gateway and
checks `C_proj_hw` from `matmul_data_mx_lut_hw.h`. That header contains a
128×128×128 FP6 contraction, 64-line A/B/C LUT banks, BF16 output, packed
output indices, and E8M0 scales. Its packed indices and scales reproduce
exactly under the explicit `radiance_header_fp6_lut_v1` projection.

The qualifier captures a PyTorch matmul through clean model2MLIR `a042643`,
checks its standard `linalg.matmul` data flow, and binds the header's actual
packed operands, scales, and LUTs to the legal E3M2 Rocket profile. The
compiler creates a data-free RV64/RoCC object. A thin C driver passes runtime
buffers, calls `mx_issue`, and compares all source-header outputs on Nicolas's
pinned Spike: **0 / 16,384 BF16**, **0 / 8,192 packed-byte**, and **0 / 512
scale** mismatches. Two independent builds have identical frontend MLIR,
bound MLIR, object, ELF, extension, Spike log, and receipt hashes.

This establishes **numerical equivalence for the source header on Rocket**.
The original program's MMIO gateway writes, scratchpad placement, polling,
and cycle behavior were not reproduced or run on a compatible MMIO model.
The [index](index.json) and [receipt](receipt.json) keep those target scopes
separate. The archive includes the payload bundle, frontend and bound MLIR,
physical commands, object, thin driver, ELF, and Spike log; the extension
binary is rebuilt from pinned RTL and its digest is in the receipt.

From a clean checkout of model2MLIR at `a042643e31366724ca0482389c4ee85a9f88e343`:

```sh
python -m tools.qualify_nicolas_ws_mmio_semantics \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt "$MX_OPT" --out-dir "$OUT_DIR"
```

The command rejects changed model2MLIR package bytes, source/header bytes,
RTL profile bytes, unsupported graphs, and mismatched output values.
