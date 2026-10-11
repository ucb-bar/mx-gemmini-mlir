# Typed MX+VPU dense attention on Spike

This archive qualifies Nicolas's pinned `attn_vpu.c` 32×64×64 kernel from
`npu-exploration` `0344bcb`. The current model2MLIR attention capture and
checked VPU/requant precursor are archived in the linked
[source baseline](../nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/README.md)
and [vector middle](../nicolas_attn_vpu_generated_middle_e9ded36_0344bcb/README.md).
The [typed program](attention.program_bound.mlir) adds explicit scale DMA,
operand DMA, QK and PV scratchpad matmuls, and four readouts to the vector
middle. Its 12 runtime buffers, source digest, quantization contract, and
target profile are checked before lowering.

`tools.compile_object` verifies the graph with `mx-gemmini-opt`, lowers all
126 MX commands and one fence, and emits one data-free RV64 RoCC
[object](typed/mx_issue.o). Its SHA-256 is
`37edc5491e0d2f03c933cc4048c043cfa7f7d14bc3af890fa32cc9bad74ccded`,
identical to the earlier [single-object physical replay](../nicolas_attn_vpu_single_object_e9ded36_0344bcb/README.md).
The [object manifest](typed/object_manifest.json) records zero allocated data
section bytes, no embedded operands or goldens, and the buffer ABI.

The [ELF](attn_vpu_typed_program.elf) links that one compiler object with the
source's deterministic data and correctness checker. Two pinned Spike runs
produced identical logs. The [archived log](spike_1.log) reports zero differing
score words, matching score/probability/requant/PV hashes, and `attn_vpu
PASSED`. The [receipt](receipt.json) records the compiler, model2MLIR, source,
RTL, toolchain, extension, object, ELF, and log digests. The source checker
still performs setup flush and final fence. This receipt qualifies this exact
source kernel and profile; it does not establish other shapes or full-model
inference.

To replay from the repository root, set `NPU_ROOT` to pinned
`npu-exploration` `0344bcb`, `RTL_ROOT` to `gemmini-mx-cleanup` `266c593`,
`RISCV_ROOT` to the RV64 toolchain, `SPIKE_EXTENSION` to Nicolas's MX Spike
extension, `MX_OPT` to the built `mx-gemmini-opt`, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_attn_compute \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --mx-opt "$MX_OPT" \
  --typed-program --out-dir "$OUT_DIR"
```

The qualifier regenerates the typed graph from the pinned frontend and source,
builds it through the public object CLI, compares its bytes against the
source-equivalent physical stream, and checks the source oracle on Spike.
