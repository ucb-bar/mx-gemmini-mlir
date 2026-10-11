# Nicolas FP8 native K-tiled DRAM loop

The pinned `matmul_tiled_fp8_128x128_dramloop_kt.c` runs two N-axis chunks,
each with two K tiles of 64. Its first K loop accumulates without storing C;
the second loop accumulates and writes that chunk's BF16 output. Each loop
provides its own A/B scale slice through LoopMatmul's scale-pointer commands.

A fresh PyTorch → model2MLIR capture, source-bound typed MX MLIR module, and
public data-free RV64 object pass **all 16,384 BF16 output comparisons** on
Nicolas's pinned Spike. The original source executable passes independently.
The [command audit](native_dram_equivalence.json) checks four loop launches,
their K-sliced scale and operand pointers, two accumulation flags, the two C
stores, and alternating B scratchpad IDs. This establishes numerical and
command-operand parity on Spike, not timing or FPGA performance.

Reproduce from compiler commit `af0b4ad`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x128_native_dram_kt2 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
