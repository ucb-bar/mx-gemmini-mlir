# Typed four-block flash attention on Spike

This archive reproduces Nicolas's `attn_flash_mid.c` output with one
compiler-generated RV64 RoCC object. The kernel has `Sq=64`, `Sk=128`,
`d=64`, and four `Bk=32` key blocks. A clean model2MLIR `a042643` capture
provides the QK→scale→softmax→PV dataflow; the pinned `npu-exploration`
`0344bcb` source and header provide the MX scale, scratchpad, and VPU
schedule. The selected RTL is `gemmini-mx-cleanup` `266c593` with VPU.

The [typed graph](flash_mid.mlir) contains pitched Q/K/V transfers,
gated weight-scale loads, managed QK/PV matmuls, online VPU softmax and
output updates, resident FP8 requantization, and BF16 output readout.
It lowers through the shared [physical command IR](physical_program.json)
to **156 MX commands and 21 fences**. The [RV64 object](mx_issue.o) carries
no operands, expected values, or data section; its [manifest](object_manifest.json)
defines ten runtime buffers. The compiler uses a **fenced serial schedule**
to establish functional parity. Nicolas's original source uses a faster
pipelined schedule; this archive makes no performance equivalence claim.

The [driver](driver.c) runs Nicolas's unmodified pipelined source for the
reference path and calls only the compiler-generated `mx_issue` for the
generated path. Both ran twice on the pinned Spike extension. Every one
of the **4,096 BF16 output values** matches bit for bit across the four
[source](source_spike_1.log) and [generated](generated_spike_1.log) logs.
The common output hash is `cd7aef4d68d1e0d9`; the source's own checker
reports `attn_flash PASSED`. The [receipt](receipt.json) pins the source,
frontend, toolchain, native verifier, object, ELFs, and logs.

To reproduce from a clean checkout, build `mx-gemmini-opt`, then set
`NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_mid_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```

This qualifies the E4M3, noncausal, four-block source shape on the pinned
MX+VPU configuration. Other flash shapes, FP4, causal masks, fused VPU
variants, and pipelined scheduling remain separate work.
