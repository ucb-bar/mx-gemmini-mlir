# Typed fused EXPSUM flash attention on Spike

This qualification selects the fused `ATTN_EXPSUM=1` path of Nicolas's
`attn_flash.c` at `npu-exploration` `0344bcb`. The pinned [wrapper](../../../tools/fixtures/nicolas_flash_mid_expsum.c)
uses the existing 64×128×64, `BK=32`, FP8 data header and the clean
model2MLIR `a042643` capture of the same QK→softmax→PV shape. The compiler
checks the wrapper, source core, frontend, data header, RTL-derived profile,
and target toolchain before lowering.

The [typed graph](flash_mid_expsum.mlir) has four `mx_gemmini.vpu_execute`
EXPSUM operations with explicit second destinations. The selected profile
must enable the EXPSUM VPU gate. The public object compiler emits
[148 MX commands and 21 fences](object/physical_program.json), eight fewer
commands than the split SUB→EXP→RSUM path. Its [RV64 object](object/mx_issue.o)
has no embedded operand or golden data and no allocated data section.

The [source](source_spike_1.log) and [compiler](generated_spike_1.log) drivers
each ran twice on pinned Spike. All **4,096 BF16 output words** match across
the four runs; the output hash is `cd7aef4d68d1e0d9`. The [receipt](receipt.json)
records the source, frontend, profile, object, ELF, simulator, and log digests.
The committed compiler reproduced the object and both ELFs byte for byte.
This is the small four-block noncausal source shape. Larger shapes and causal
masking need separate qualification.

To reproduce, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_mid_expsum_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
