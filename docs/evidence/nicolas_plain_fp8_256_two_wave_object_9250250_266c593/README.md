# Nicolas FP8 128×128×256 with compiler-scheduled K waves

Commit `9250250` captured Nicolas's 128×128×256 FP8 matrix shape from PyTorch
using current model2MLIR `e9ded36`, bound the exact packed operands, E8M0
scales, and BF16 golden from `matmul_tiled_fp8_128x128x256.c` and its checked-in
header, and compiled a data-free RV64 RoCC object through `tools.compile_object`.
The physical plan selects **two K waves of 128** and accumulates the second
wave before reading out. Pinned Spike compared **all 16,384 BF16 outputs**
against Nicolas's source header and reported zero mismatches.

This is numerical parity for the matrix result using a compiler-selected
two-wave schedule. It does not reproduce the C program's exact instruction
ordering, cache behavior, or performance counters. The selected profile is
the MX-only `MxGemminiRocketConfig` at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. The prototype and replay
from the committed compiler produced identical model2MLIR, bound MLIR,
physical program, object, ELF, and Spike-log hashes.

The archive contains the full source payload, typed MLIR, generated issuer,
physical plan, object, ELF, Spike result, and a receipt pinning all major
source/tool/artifact digests.

Reproduce from this compiler commit with the pinned RTL and submodules,
model2MLIR/MXQuant revisions in `receipt.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x256 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
