# Nicolas FP6 128×128×512 from typed MLIR to a public MX object

Commit `95fc6d5` captured a 128×512 by 512×128 PyTorch matmul with current
model2MLIR `e9ded36`, then bound Nicolas's checked-in packed FP6 operands,
E8M0 scales, and complete 64-line A/B/C LUT banks. The source header's first
A/B LUT lines select the frontend's structural FP6 codebook; the payload bundle
supplies every actual hardware LUT line. `tools.compile_object` lowered the
typed contraction to a data-free RV64 RoCC issuer. A driver that only passes
source buffers, fences, and checks results linked the object with Nicolas's
bare-metal support. Pinned Spike matched **all 16,384 BF16 values** in
`matmul_fp6_128x128x512.h`.

The selected configuration is the MX-only `MxE3M2OnlyGemminiRocketConfig` at
RTL revision `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, using the
compiler's serial Spike schedule. The result qualifies this source matrix
operation on the pinned model. It does not establish the alternating-buffer
RTL schedule or reproduce the C program's performance counters. An earlier
CLI run and the replay from the committed compiler produced identical
frontend policy, MLIR, object, physical program, ELF, and Spike-log hashes.

`receipt.json` pins the source/header, frontend, profile, compiler, object,
ELF, and simulator digests. `bundle/` contains the source arrays and golden;
`payload_bound.mlir`, `object/`, and `run/` preserve the compiler and execution
artifacts.

Reproduce from this compiler commit with the pinned RTL and its submodules,
MXQuant revision in the receipt, a native `mx-gemmini-opt`, and RISC-V
GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp6_128x128x512 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
