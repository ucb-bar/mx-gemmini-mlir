<a id="nicolas-four-block-flash-attention-source-oracle"></a>

# Reference four-block flash attention source oracle

This archive pins the unmodified `attn_flash_mid.c` wrapper and shared
`attn_flash.c` from `npu-exploration` `0344bcb`. The kernel uses the
[64×128×64 generated data header](../nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/README.md)
with `BK=32`: four key blocks, resident K/V, a bank-split scratchpad, online
softmax on the VPU, resident requantization, and pipelined QK/PV issue.
The source folds `1/√64` into Q's E8M0 scales.

The [original source ELF](source.elf) ran twice on the reference implementation’s pinned Spike
model with identical [logs](source_spike.log). Its complete output hash is
`cd7aef4d68d1e0d9`; the source checker reports a match and `attn_flash
PASSED`. The [expectation header](attn_flash_mid_expect.h) and
[receipt](receipt.json) pin that oracle, the source and data hashes,
selected profile, RTL, toolchain, Spike extension, and ELF. A fresh replay
with `tools.qualify_nicolas_flash_mid_source` reproduced the **same ELF and
Spike log SHA-256**, recorded in its [replay receipt](replay_receipt.json).

The clean model2MLIR `a042643` capture in the linked data archive gives the
equivalent full QK→softmax→PV tensor dataflow. The source's online softmax,
managed scale-half protocol, block buffers, and overlapping matrix/VPU issue
require a separate typed schedule. The [compiler-generated serial schedule](../nicolas_flash_mid_typed_a042643_0344bcb/README.md)
now matches every BF16 output on Spike. This archive remains the original
source oracle; its ELF does not contain compiler-generated MX commands.

From the repository root, set `NPU_ROOT` to pinned `npu-exploration`
`0344bcb`, `RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to
the RV64 toolchain, `SPIKE_EXTENSION` to the reference implementation’s MX Spike extension, and
`OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_flash_mid_source \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --out-dir "$OUT_DIR"
```
