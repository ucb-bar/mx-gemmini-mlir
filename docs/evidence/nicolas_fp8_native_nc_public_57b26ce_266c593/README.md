# Nicolas FP8 native DRAM column chunks

The pinned `matmul_tiled_fp8_128x128_dramloop_nc.c` and `...nc4.c` programs
split a 128×128×128 FP8 matrix multiplication into two and four N-axis loop
launches. A is loaded once and stays resident; B uses alternating scratchpad
banks; the native loop writes each BF16 C slice directly to DRAM. The compiler
captures `torch.matmul` through model2MLIR, binds the exact source arrays to
typed MX MLIR, and emits data-free RV64 objects with those loop schedules.

For **each** source program, the original executable and generated object
independently pass **all 16,384 BF16 output comparisons** on Nicolas's pinned
Spike. The [two-column audit](nc2/native_dram_equivalence.json) and
[four-column audit](nc4/native_dram_equivalence.json) check loop bounds,
runtime pointer offsets, B bank IDs, repeated A scales, B scale slices, and
the absence of explicit operand/output DMA. There is no fence between loop
launches. The result establishes numerical and command-operand parity on
Spike, not cycle counts, overlap timing, or FPGA performance.

Reproduce from compiler commit `57b26ce`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
for chunks in 2 4; do
  "$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
    --case "fp8_128x128x128_native_dram_nc${chunks}" \
    --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
    --rtl-root "$MX_RTL_ROOT" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
    --out-dir "$OUT_ROOT/nc${chunks}"
done
```

The qualifier refuses to overwrite existing output directories.
