# Nicolas packed FP4 requantized output through a public MX object

Commit `5657fb6` captured a 64³ PyTorch matmul with model2MLIR `e9ded36`,
bound Nicolas's checked-in `matmul_tiled_fp4_64x64_requant.c` operands and
`matmul_fp4_64x64.h` golden, selected typed FP4 quantized readout, and compiled
a data-free RV64 RoCC issuer through `tools.compile_object`. Pinned Spike
matched the source header's **2,048 packed FP4 output bytes and 128 E8M0
scales**, with zero mismatches in both arrays.

Nicolas's header stores two FP4 codes per byte, paired across even and odd
matrix rows. Its explicit `C_scales_out` is also read from the header. The
qualifier checks both arrays against an independent BF16-to-FP4 reference
before compiling. The generated driver supplies source buffers, fences, and
compares the complete outputs; it contains no replacement MX command
schedule. The prototype and committed replay produced identical model2MLIR,
bound MLIR, physical program, object, ELF, and Spike-log hashes.

This qualifies the matrix and packed requantized outputs of this source test
on the MX-only `MxGemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. It does not reproduce the
C program's performance instrumentation or other FP4 source variants. The
archive includes the exact source bundle, typed MLIR, generated issuer,
physical program, object, ELF, Spike result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp4_64x64x64_requant \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
