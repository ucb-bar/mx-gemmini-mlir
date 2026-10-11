# Generated MX attention commands through Spike

This extends the [compute-stage replay](../nicolas_attn_vpu_generated_compute_e9ded36_0344bcb/README.md)
for Nicolas's `attn_vpu.c` at `npu-exploration` `0344bcb`. The selected profile
is the DIM16 MX+VPU Rocket config from `gemmini-mx-cleanup` `266c593`, and the
bound vector middle comes from the model2MLIR `e9ded36` capture.

The compiler issues source-equivalent commands for the Q/Kᵀ/V E8M0 scale
loads, three operand DMAs, QK, seven VPU steps, resident E4M3 requant, PV,
and all four scratchpad readouts. Every issuer object has zero allocated data
bytes. The linked ELF uses the original source's deterministic input header
and correctness checker. It matched all 512 score words against the mesh
golden, plus the score, probability, quantized probability, and output hashes
from the unmodified source. Two Spike logs were byte identical.

The source C still calls `gemmini_flush`, `gemmini_fence`, and the host checker.
QK/PV and I/O packets are bounded source-specific physical lowering; only
the VPU/requant middle is currently driven by typed MLIR. The next gate is a
single typed graph for both matrices and the I/O stages, lowered to one
linkable object with a declared runtime ABI.

From the repo root, set `NPU_ROOT` to the pinned `npu-exploration` checkout,
`RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX extension, and `OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_attn_compute \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --generated-io --out-dir "$OUT_DIR"
```

The [receipt](receipt.json) pins the source, model graph, compiler, RTL,
toolchain, issuer objects, ELF, and Spike log. `--mx-opt` can also verify the
typed vector stage natively during the build.
