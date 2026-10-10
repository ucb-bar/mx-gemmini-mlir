# Nicolas FP4 128×128×512 requantized output through a public MX object

Commit `03490a5` captured a 128×128×512 PyTorch matmul with model2MLIR
`e9ded36`, bound Nicolas's checked-in
`matmul_tiled_fp4_128x128x512_requant.c` operands and
`matmul_fp4_128x128x512.h` golden, selected typed FP4 quantized readout, and
compiled a data-free RV64 RoCC issuer through `tools.compile_object`. Pinned
Spike matched **all 8,192 packed FP4 output bytes and 512 E8M0 scales** in
the source header, with zero mismatches.

The header's `C_out[64][128]` pairs even and odd matrix rows in each byte.
Its explicit `C_scales_out[128][4]` is the scale golden. The qualifier checks
both against an independent BF16-to-FP4 reference before compiling. The
generated driver supplies source arrays, fences, and compares the complete
outputs; it contains no replacement MX command schedule. The prototype and
replay from the committed compiler produced identical model2MLIR, bound
MLIR, physical program, object, ELF, and Spike-log hashes.

This qualifies this source test's matrix and packed requantized results on
the MX-only `MxGemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. It extends the direct FP4
quantized shape beyond 64³; it does not establish every shape, other mesh
geometry, or performance parity. The archive includes the exact source
bundle, typed MLIR, generated issuer, physical program, object, ELF, Spike
result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp4_128x128x512_requant \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
