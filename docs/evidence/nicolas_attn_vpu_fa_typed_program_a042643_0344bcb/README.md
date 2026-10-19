# Typed 64×256×128 dense MX+VPU attention on Spike

the reference implementation’s [unmodified `attn_vpu_fa` source](../nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/README.md)
provides E4M3/E8M0 operands, a mesh score golden, and four stage hashes.
A clean model2MLIR `a042643` capture supplies QK, scale by `1/√128`,
softmax, and PV dataflow. The [bound vector graph](attention.vector_bound.mlir)
and [binding](vector_binding.json) connect that frontend to the source VPU
and resident requant schedule. The [typed MX program](attention.program_bound.mlir)
derives scale and operand transfers, two differently shaped scratchpad
matmuls, and readouts from the 64×256×128 source shape and target profile.

`tools.compile_object` verifies the graph with `mx-gemmini-opt` and emits
one data-free RV64 RoCC [object](object/mx_issue.o): 702 MX commands and one
fence. Its [manifest](object/object_manifest.json) records the 12-buffer ABI
and zero allocated data section bytes. The object SHA-256 is
`7d6d6dd973634bdf8f8da24b92e9e6ed62ce3e5af3e9ca7c17741dfafc7fcd37`.

The [ELF](attn_vpu_typed_program.elf) links that object with the source's
deterministic data and correctness checker. Two pinned Spike runs had
identical logs. The [archived log](spike_1.log) reports zero differing score
words out of 4,096 four-element words, matching score, probability,
requantized probability, and PV hashes, and `attn_vpu PASSED`. The checker
still issues setup flush and final fences; all dense attention kernel
commands are in the compiler object. The [receipt](receipt.json) pins
compiler, source, model2MLIR, profile, RTL, toolchain, Spike extension,
object, ELF, and log digests. This is the dense `attn_vpu_fa.c` shape; it
does not qualify the separate blocked `attn_flash.c` family or full models.

From the repository root, set `NPU_ROOT` to pinned `npu-exploration`
`0344bcb`, `RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to
the RV64 toolchain, `SPIKE_EXTENSION` to the reference implementation’s MX Spike extension,
`MX_OPT` to the built `mx-gemmini-opt`, and `OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_dense_attention \
  --variant fa --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --mx-opt "$MX_OPT" \
  --out-dir "$OUT_DIR"
```

The same command with `--variant base` or `--variant mid` reproduces the
smaller source shapes and their archived object SHA-256 digests.
