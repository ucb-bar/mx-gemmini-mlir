# Nicolas FP6 128³ with repeated LUT codes

Nicolas's pinned `matmul_tiled_fp6_128x128.c` uses 64 spatial LUT lines per
operand, with repeated E3M2 codes in every A and B line. Some source nibble
indices therefore differ from the canonical lowest-index encoding while
decoding to the same code. The compiler binds the **original** packed source
bytes and all original A/B/C LUT lines; it does not replace the source indices
with canonical aliases.

The PyTorch → model2MLIR capture supplies the contraction structure. MXQuant
requires distinct codes in a capture codebook, so its archived
`source_line0_policy.yaml` uses a deterministic unique witness derived from
source line zero. That witness is **not** the execution LUT. The source payload
manifest binds the actual checked-in arrays and LUT banks to typed MX MLIR.

The original C source and the generated public data-free RV64 object pass
independently on Nicolas's pinned Spike. The compiled program matches all
**16,384 BF16 source-golden outputs** under the `MxE3M2OnlyGemminiRocketConfig`
profile. This qualifies the numerical serial schedule, not alternating-buffer
timing or FPGA performance.

Reproduce from compiler commit `6c6b27a9cfe7faf88326d7ab26eeb00c1e37d841`,
RTL `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp6_128x128x128 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
