# Typed FP4 blocked flash attention on Spike

This qualification uses Nicolas's pinned `attn_flash.c` core (`npu-exploration`
`0344bcb`) with a four-line [FP4 wrapper](../../../tools/fixtures/nicolas_flash_mid_fp4.c).
The source generator `gen_attn_fp4.py` produced the [packed Q/K/V data and
E8M0 scales](attn_vpu_mid_fp4.h.gz) from random seed 0 using `--sq 64 --sk 128
--d 64 --bk 32 --tag _mid_fp4`. Its independent bit-exact golden is
`f0caacc02eab4345`. This is a small FP4 configuration of the source core,
not a claim that Nicolas's named 2048-key FP4 experiments compile.

The [typed graph](flash_mid_fp4.mlir) uses the clean model2MLIR `a042643`
capture of the same QK→softmax→PV shape and binds it to this FP4 data header.
It configures the quad mesh before Q-scale DMA, loads packed operand tiles,
issues eight managed FP4 matmuls, requantizes four probability blocks to
resident E2M1, and runs the online BF16 VPU updates. The public object
compiler lowers it to [154 MX commands and 21 fences](object/physical_program.json).
The [RV64 object](object/mx_issue.o) has no operand, expected-output, or
allocated data section.

The [source](source_spike_1.log) and [compiler](generated_spike_1.log) paths
both ran twice on pinned Spike. All **4,096 BF16 output words** match bit for
bit across the four runs, with hash `f0caacc02eab4345`. The
[receipt](receipt.json) records the source, frontend, profile, generator,
toolchain, simulator, object, ELF, and log digests. A replay from the committed
compiler reproduced the object and both ELFs byte for byte.

To reproduce, set `NPU_ROOT` to the `npu-exploration` `0344bcb` checkout,
`RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64
toolchain, `SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR`
to a fresh directory:

```sh
python -m tools.qualify_nicolas_flash_mid_fp4_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```

The qualification reads the archived header and checks its digest. The
generator recipe above explains its provenance; running the qualification
does not require PyTorch. Causal masking, streamed K/V, fused VPU passes,
larger FP4 shapes, and pipelined compiler scheduling have separate gates.
