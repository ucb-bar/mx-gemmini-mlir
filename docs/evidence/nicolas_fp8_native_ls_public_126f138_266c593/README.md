# Nicolas FP8 loop-managed native scales

The pinned `matmul_tiled_fp8_128x128_dramloop_ls.c` and `...ls4.c` use two
and four native N-axis LoopMatmul launches. Each launch supplies its own
A/B E8M0 scale pointers and row pitches through the loop-managed MX scale
commands. The compiler captures `torch.matmul` through model2MLIR, binds the
exact source arrays to typed MX MLIR, and emits data-free RV64 objects.

For **each** program, the original C executable and generated object
independently pass **all 16,384 BF16 output comparisons** on Nicolas's pinned
Spike. The [two-loop audit](ls2/native_dram_equivalence.json) and
[four-loop audit](ls4/native_dram_equivalence.json) check the scale pointer
commands (`funct` 31/32), loop bounds, operand/output pointer offsets, and
alternating B scratchpad IDs. The generated command stream has no standalone
scale upload, scale-selector command, or fence between loop launches. This
establishes numerical and command-operand parity on Spike, not timing or FPGA
performance.

Reproduce from compiler commit `126f138`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
for chunks in 2 4; do
  "$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
    --case "fp8_128x128x128_native_dram_ls${chunks}" \
    --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
    --rtl-root "$MX_RTL_ROOT" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
    --out-dir "$OUT_ROOT/ls${chunks}"
done
```

The qualifier refuses to overwrite existing output directories.
