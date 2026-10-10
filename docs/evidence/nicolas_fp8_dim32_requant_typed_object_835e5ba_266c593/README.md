# Nicolas FP8 DIM32 requantized output through a public MX object

Commit `835e5ba` captured a 64³ PyTorch matmul with model2MLIR `e9ded36`,
bound Nicolas's checked-in `matmul_tiled_fp8_64x64_requant_dim32.c` operands
and `matmul_fp8_64x64_dim32.h` golden, selected typed FP8 quantized readout,
and compiled a data-free RV64 RoCC issuer through `tools.compile_object`.
Pinned Spike with the **DIM32 extension** matched all **4,096 FP8 output codes
and 128 E8M0 scales** in the source header, with zero mismatches.

The qualifier checks both source arrays against an independent BF16-to-FP8
reference before compiling. It builds the Spike extension with
`GEMMINI_DIM=32` from the selected profile and runs `gemmini_dim32`. A prior
probe with the default DIM16 extension reported thousands of mismatches;
that probe demonstrated why the profile must select the simulator geometry.
The generated driver supplies source buffers, fences, and compares the
complete outputs; it contains no replacement MX command schedule. The
successful prototype and committed replay produced identical model2MLIR,
bound MLIR, physical program, object, ELF, extension, and Spike-log hashes.

This qualifies this source test's matrix and requantized results on the
MX-only `MxDim32GemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. It does not establish
every DIM32 source variant, RTL timing, or performance parity. The archive
includes the exact source bundle, typed MLIR, generated issuer, physical
program, object, ELF, Spike result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_64x64x64_requant_dim32 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim32GemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
