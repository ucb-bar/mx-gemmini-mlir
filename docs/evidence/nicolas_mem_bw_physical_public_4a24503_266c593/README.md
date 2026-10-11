# Nicolas's MX memory benchmark through generated DMA commands

Compiler commit `4a24503` derives four reusable physical command phases from
the pinned MX-only profile: 64-byte A matrix loads, 16-byte B matrix loads,
an E8M0 scale load, and a 16-byte scratchpad readout. The compiler emits a
data-free RV64 RoCC object with setup and phase entry points. A driver derived
from Nicolas's `mx_mem_bw.c` keeps his counters, phase reporting, and spot
check while calling the generated object at each accelerator issue site.

The original source and generated driver both pass on Nicolas's pinned Spike.
All seven phases report the source's byte and request counts, and an
instrumented source run and generated driver produce **identical complete
16,384-byte readout buffers**. The full buffers are archived as hex in the
two Spike logs; `receipt.json` pins their SHA-256 digest. The emitted object
contains no operands or golden data.

This is a **physical command IR** replay of a memory microbenchmark; its
transfer phases are not yet represented by dedicated typed MX MLIR ops.
Spike cycle and performance-counter values are reported but not used as a
parity criterion. The `matmul_fp8_128x128.h` source data and
`MxGemminiRocketConfig` at RTL `266c593` are pinned in the receipt.

Reproduce from commit `4a24503` with the pinned RTL/submodules and RISC-V
GCC/Spike; the command requires a fresh output directory:

```sh
python -m tools.qualify_nicolas_mem_bw \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --out-dir "$OUT_DIR"
```
