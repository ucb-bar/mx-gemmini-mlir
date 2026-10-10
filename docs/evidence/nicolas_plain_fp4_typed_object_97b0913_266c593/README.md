# Nicolas FP4 64³ from typed MLIR to a public MX object

Commit `97b0913` captured a 64×64 PyTorch matmul with current model2MLIR
`e9ded36`, then bound Nicolas's checked-in direct FP4 packed operands, E8M0
scales, and BF16 golden from `matmul_tiled_fp4_64x64.c` and its data header.
The source header uses literal C array dimensions and the hardware's packed
activation layout. `tools.compile_object` produced a data-free RV64 RoCC
issuer; a driver only supplied those source arrays, fenced, and compared all
4,096 BF16 outputs. Pinned Spike reported **zero mismatches**.

The selected target is the MX-only `MxGemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. This qualifies the
source matrix result, not its performance counters or other FP4 source
programs. The prototype and replay from the committed compiler had identical
model2MLIR, handoff, bound MLIR, object, physical program, ELF, and Spike-log
SHA-256 hashes.

The archive includes the full source payload bundle, typed MLIR, generated
issuer and object, ELF, full-output Spike result, and `receipt.json` with
source, tool, profile, and artifact hashes.

Reproduce from this compiler commit with the pinned RTL and submodules,
model2MLIR/MXQuant revisions in the receipt, a native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp4_64x64x64 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
