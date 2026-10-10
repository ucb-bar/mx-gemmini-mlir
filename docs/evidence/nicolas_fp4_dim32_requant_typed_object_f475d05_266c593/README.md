# Nicolas FP4 DIM32 requantized output through a public MX object

Commit `f475d05` captured a 128³ PyTorch matmul with model2MLIR `e9ded36`,
bound Nicolas's checked-in `matmul_tiled_fp4_128x128_requant_dim32.c`
operands and `matmul_fp4_128x128_dim32.h` golden, selected typed FP4
quantized readout, and compiled a data-free RV64 RoCC issuer through
`tools.compile_object`. Pinned Spike with the **DIM32 extension** matched
all **8,192 packed FP4 output bytes and 512 E8M0 scales** in the source
header, with zero mismatches.

The header stores pairs of output rows in `C_out[64][128]` and explicit
`C_scales_out[128][4]`. The qualifier checks both against an independent
BF16-to-FP4 reference before compiling. It builds the Spike extension with
`GEMMINI_DIM=32` from the selected profile and runs `gemmini_dim32`. The
generated driver supplies source buffers, fences, and compares the complete
outputs; it contains no replacement MX command schedule. The successful
prototype and committed replay produced identical model2MLIR, bound MLIR,
physical program, object, ELF, extension, and Spike-log hashes.

This qualifies this source test's matrix and requantized results on the
MX-only `MxDim32GemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. Together with the DIM32
FP8 replay, it establishes two direct precision modes on this mesh; other
DIM32 modes, RTL timing, and performance parity remain unqualified. The
archive includes the exact source bundle, typed MLIR, generated issuer,
physical program, object, ELF, Spike result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp4_128x128x128_requant_dim32 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim32GemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
