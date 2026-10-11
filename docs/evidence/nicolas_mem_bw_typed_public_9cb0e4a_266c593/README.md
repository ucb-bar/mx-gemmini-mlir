# Nicolas's MX memory benchmark from typed MLIR to RV64

Compiler commit `9cb0e4a` defines typed `mx_gemmini.memory_setup`,
`dma_matrix`, `load_scales`, and `spad_mvout_linear` operations. The archived
`typed_memory.mlir` binds Nicolas's source SHA and the pinned MX-only profile.
Native `mx-gemmini-opt` verifies the operations, then profile-aware lowering
selects 64-byte A loads, 16-byte B loads, an E8M0 scale load, and a 16-byte
scratchpad readout. The compiler emits a data-free RV64 RoCC object with setup
and phase entry points. A driver derived from Nicolas's `mx_mem_bw.c` keeps his
counters, phase reporting, and spot check while calling the generated object
at each accelerator issue site.

The original source and generated driver both pass on Nicolas's pinned Spike.
All seven phases report the source's byte and request counts, and an
instrumented source run and generated driver produce **identical complete
16,384-byte readout buffers**. The full buffers are archived as hex in the
two Spike logs; `receipt.json` pins their SHA-256 digest. The emitted object
contains no operands or golden data.

The typed route emits issuer C and object bytes **identical** to the earlier
[physical command replay](../nicolas_mem_bw_physical_public_4a24503_266c593/README.md).
Spike cycle and performance-counter values are reported but are not a parity
criterion. The `matmul_fp8_128x128.h` source data and
`MxGemminiRocketConfig` at RTL `266c593` are pinned in the receipt.

Reproduce from commit `9cb0e4a` with the pinned RTL/submodules, native
`mx-gemmini-opt`, and RISC-V GCC/Spike; the command requires a fresh output
directory:

```sh
python -m tools.qualify_nicolas_mem_bw \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```
