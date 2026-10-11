# Typed 64×128×64 MX+VPU attention on Spike

Nicolas's [unmodified mid-shape source](../nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/README.md)
provides the E4M3/E8M0 operand data, mesh score golden, and four stage hashes.
A clean model2MLIR `a042643` capture supplies QK, scale, softmax, and PV
dataflow. The [bound vector graph](attention.vector_bound.mlir) and
[binding](vector_binding.json) connect that frontend to the source VPU and
resident requant schedule. The [typed MX program](attention.program_bound.mlir)
adds profile-derived scale/operand transfers, two differently shaped
scratchpad matmuls, and readouts.

`tools.compile_object` verifies this graph with `mx-gemmini-opt` and lowers it
to one data-free RV64 RoCC [object](object/mx_issue.o): 302 MX commands and
one fence. Its [manifest](object/object_manifest.json) records the 12-buffer
ABI and zero allocated data section bytes. The object SHA-256 is
`0a8f8f1b541c02aec1a2bbe42f826dbb7931ac4a3b6ade3c2d6b3809b582f631`.

The [ELF](attn_vpu_typed_program.elf) links that object with the source's
deterministic data and correctness checker. Two pinned Spike runs had
identical logs. The [archived log](spike_1.log) reports zero differing score
words out of 2,048 four-element words, matching score, probability,
requantized probability, and PV hashes, and `attn_vpu PASSED`. The checker
still issues setup flush and final fences; all attention kernel commands are
in the compiler object. The [receipt](receipt.json) pins compiler, source,
model2MLIR, profile, RTL, toolchain, Spike extension, object, ELF, and log
digests. This qualifies the 64×128×64 source shape, not a full model.

From the repository root, set `NPU_ROOT` to pinned `npu-exploration`
`0344bcb`, `RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to
the RV64 toolchain, `SPIKE_EXTENSION` to Nicolas's MX Spike extension,
`MX_OPT` to the built `mx-gemmini-opt`, and `OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_dense_attention \
  --variant mid --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --mx-opt "$MX_OPT" \
  --out-dir "$OUT_DIR"
```

The same command with `--variant base` replays the earlier 32×64×64 source
and reproduces its original object SHA-256.
