# Nicolas FP8 native DRAM-loop replay

The pinned `matmul_tiled_fp8_128x128_dramloop.c` loads A and B and stores BF16
C through LoopMatmul's native DRAM path. This compiler replay captures
`torch.matmul` through model2MLIR, binds the exact source operand and scale
arrays to typed MX MLIR, and emits a data-free RV64 object through the public
object compiler. The original C program and the generated object each pass
**all 16,384 BF16 output comparisons** against the checked-in golden on
Nicolas's pinned Spike.

The [physical-command audit](native_dram_equivalence.json) checks the loop
bounds, A/B/C runtime pointers, row strides, scratchpad buffer IDs, and scale
byte placement against the source. The generated object contains no explicit
operand or output DMA commands; the loop performs those transfers. It loads
the same contiguous E8M0 scale bytes using 2D scale commands, whereas the C
source uses flat scale uploads. This is numerical and command-operand parity
on Spike. Cycle counts, overlap, and FPGA behavior are not qualified here.

Reproduce from compiler commit `1ef6f85`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x128_native_dram \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
