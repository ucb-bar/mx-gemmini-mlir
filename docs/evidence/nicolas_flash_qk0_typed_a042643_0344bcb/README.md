<a id="typed-first-block-of-nicolass-flash-attention"></a>

# Typed first block of the reference implementation’s flash attention

This archive covers QK block 0 of the reference implementation’s four-block `attn_flash_mid.c`
(`Sq=64, Sk=128, d=64, Bk=32`). The clean model2MLIR `a042643` capture
identifies the full QK→scale→softmax→PV dataflow. The pinned source wrapper,
core, and generated data header bind that capture to the first 64×32 QK tile.
The source folds `1/√64` into Q's E8M0 scales; the generated object receives
those folded scales as a runtime buffer.

The [typed graph](flash_qk0.mlir) lowers to [35 RoCC commands and one fence](physical_program.json):
folded Q scale DMA, Q and pitched Kᵀ operand DMA, FP8/BF16 mesh setup, gated
weight-scale DMA, managed scale-half QK matmul, and BF16 score readout. The
[linkable RV64 object](mx_issue.o) contains no operand or golden data. Its
[manifest](object_manifest.json) records the six runtime buffers and tool hashes.

The [source driver](driver.c) invokes the reference implementation’s original `issue_mm` for QK0.
The same driver with `GENERATED` invokes only the compiler's `mx_issue` object.
Both ran twice on the reference implementation’s pinned Spike model. All **2,048 BF16 score words**
match bit for bit in the four [source](source_spike_1.log) and
[generated](generated_spike_1.log) logs. The common score hash is
`52bca547381dac69`. The [receipt](receipt.json) pins both ELFs, MLIR,
object, and logs.

To reproduce from a clean checkout, build `mx-gemmini-opt`, then set
`NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to the reference implementation’s MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_qk0 \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```

This is one executable block of the blocked kernel. The other three QK blocks,
online softmax, requantization, PV blocks, output updates, and the pipelined
schedule still require typed lowering and whole-kernel Spike parity.
