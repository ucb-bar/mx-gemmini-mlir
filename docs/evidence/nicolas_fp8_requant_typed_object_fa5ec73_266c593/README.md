# Nicolas FP8 requantized output through a public MX object

Commit `fa5ec73` captured a 128³ PyTorch matmul with current model2MLIR
`e9ded36`, bound Nicolas's checked-in `matmul_tiled_fp8_128x128_requant.c`
operands and `matmul_fp8_128x128.h` golden, selected typed FP8 quantized
readout, and compiled a data-free RV64 RoCC issuer through
`tools.compile_object`. Pinned Spike matched the source header's **16,384 FP8
codes and 512 E8M0 scales**, with zero mismatches in both arrays.

The qualifier first checks that the header's `C_out` and `C_scales_out` equal
an independent BF16-to-FP8 reference for this target. It rejects a source
whose quantized convention differs rather than silently comparing to a
derived golden. The generated driver only passes source buffers, fences, and
compares both complete outputs; it contains no replacement MX command
schedule. The prototype and committed replay produced identical model2MLIR,
bound MLIR, physical program, object, ELF, and Spike-log hashes.

This qualifies this source test's matrix and requantized outputs on the
MX-only `MxGemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. It does not reproduce the
C program's performance instrumentation or establish every quantized source
variant. The archive includes the exact source bundle, typed MLIR, generated
issuer, physical program, object, ELF, Spike result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x128_requant \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
