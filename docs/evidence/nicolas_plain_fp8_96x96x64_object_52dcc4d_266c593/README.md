# Nicolas FP8 96×96×64 from typed MLIR to a public MX object

Commit `52dcc4d` captured this non-square FP8 matrix shape from PyTorch with
current model2MLIR `e9ded36`, bound Nicolas's checked-in packed operands,
E8M0 scales, and BF16 golden from `matmul_tiled_fp8_96x96x64.c` and its header,
then compiled a data-free RV64 RoCC issuer through `tools.compile_object`.
Pinned Spike matched **all 9,216 BF16 outputs** against the source header.

This qualifies the matrix result for the MX-only `MxGemminiRocketConfig` at
RTL revision `266c593f2cb51d7e3fe83fc0317072b585ac3c52`. It does not
reproduce the C program's performance counters or qualify all rectangular
shapes. A second run from the committed compiler reproduced the prototype's
model2MLIR, bound MLIR, physical program, object, ELF, and Spike-log hashes.
The archive includes the source bundle, typed MLIR, generated issuer, physical
program, ELF, Spike result, and digest receipt.

Reproduce from this compiler commit with pinned RTL/submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_96x96x64 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
