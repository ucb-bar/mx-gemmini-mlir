# Nicolas FP8 native-loop store ReLU

The pinned `matmul_tiled_fp8_128x128_dramloop_relu.c` runs two N chunks with
loop-managed scales and enables ReLU in the BF16 store configuration. Its
reference is the source header's BF16 golden with every negative-sign word
replaced by zero.

A fresh PyTorch → model2MLIR contraction, source-bound typed MX module, and
public data-free RV64 object pass **all 16,384 ReLU-adjusted BF16 output
comparisons** on Nicolas's pinned Spike. The original C source passes
independently. The [command audit](native_dram_equivalence.json) checks both
native loops, per-loop scales, and the store configuration's ReLU bit. The
physical receipt hashes the complete derived expected output; 8,492 source
words have the sign bit set. This establishes numerical and command-operand
parity, not cycle or FPGA performance parity.

Reproduce from compiler commit `c7292768d0184f06b82dcfc87a3419a8a4831a8f`,
RTL `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x128_native_dram_relu \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
