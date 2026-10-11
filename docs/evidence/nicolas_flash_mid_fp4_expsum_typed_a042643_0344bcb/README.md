# Typed FP4 flash attention with fused EXPSUM on Spike

This qualification combines two paths in Nicolas's pinned `attn_flash.c`
core (`npu-exploration` `0344bcb`): FP4 E2M1 operands and fused VPU EXPSUM.
The [wrapper](../../../tools/fixtures/nicolas_flash_mid_fp4_expsum.c)
selects `BK=32` and `ATTN_EXPSUM=1`. It uses the same [packed source
data](attn_vpu_mid_fp4.h.gz) generated for the split FP4 flash qualification
and the same clean model2MLIR `a042643` QK→softmax→PV capture. The compiler
checks the source core, wrapper, data, frontend, RTL-derived profile, and
toolchain before issuing commands.

The [typed graph](flash_mid_fp4_expsum.mlir) loads packed FP4 Q/K/V, issues
eight managed FP4 matmuls, requantizes probability blocks to resident E2M1,
and uses four fused EXPSUM operations with explicit second destinations. The
profile must enable the EXPSUM VPU gate. The public compiler emits a
[data-free RV64 object](object/mx_issue.o) with 146 MX commands and 21 fences,
eight fewer commands than the split FP4 path. The object has no allocated data
section.

The [source](source_spike_1.log) and [compiler](generated_spike_1.log) drivers
each ran twice on pinned Spike. All **4,096 BF16 output words** match across
the four runs; the source hash is `f0caacc02eab4345`. The [receipt](receipt.json)
records the source, profile, frontend, object, ELF, simulator, and log digests.
A replay from committed compiler code reproduced the object and both ELFs byte
for byte. This is the small noncausal 64×128×64 configuration; larger shapes
and causal masking need their own qualification.

To reproduce, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_mid_fp4_typed \
  --softmax-mode expsum --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
