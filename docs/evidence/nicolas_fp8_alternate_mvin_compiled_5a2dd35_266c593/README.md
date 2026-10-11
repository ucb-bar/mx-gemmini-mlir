# Nicolas alternate FP8 32³ transfer source

The pinned `matmul_tiled_fp8.c` source uses a different loop nest description
from `matmul_tiled_fp8_32x32x32.c` for B transfers, while sharing the same
checked-in 32³ operand and golden header. Both the original source executable
and a fresh PyTorch → model2MLIR → typed MX MLIR → public RV64 object replay
pass all **1,024 BF16 output comparisons** on Nicolas's pinned Spike.

The [transfer equivalence audit](transfer_equivalence.json) evaluates the
source's A and B address formulas and compares every resulting DRAM byte offset
and scratchpad row with the compiler's physical command stream. At this shape,
both loop nests flatten to the same transfers. The generated object is
byte-identical to the [already qualified 32³ object](../nicolas_direct_matrix_suite_9df3384_266c593/fp8_32x32x32/object/mx_issue.o).
This is a source-program numerical and transfer equivalence result on Spike;
it does not establish cache behavior, cycle count, or FPGA performance.

Reproduce with compiler commit `5a2dd35`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_32x32x32_alternate_mvin \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The command refuses to overwrite an existing output directory.
