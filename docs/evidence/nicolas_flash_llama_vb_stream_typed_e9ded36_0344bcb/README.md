# Typed streamed flash with uneven key blocks on Spike

the reference implementation’s unmodified `attn_flash_llama_vb.c` wrapper partitions 2,048 keys
into **64 + 7×256 + 192**. It uses the same synthetic operands and
model2MLIR `e9ded36` QK→scale→softmax→PV capture as the [uniform 256-key
case](../nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb/README.md).
Its online softmax rounds at different block boundaries, so the source
output hash is `747ca99b19c32066`.

The [typed graph](flash_llama_vb_stream.mlir) binds the source wrapper,
block lengths, key offsets, K×2/V×3 ring placement, gated scale slices,
and each block's QK, VPU softmax, requant, PV, and output update. The first
block selects the source's 128-byte BF16 store stride. The public compiler
emits [508 MX commands and 56 fences](object/physical_program.json) in a
[data-free RV64 object](object/mx_issue.o).

The [source](source_spike_1.log) and [compiler](generated_spike_1.log)
drivers each ran twice on the pinned MX Spike extension. **All 4,096 BF16
output words match bit for bit** across the four runs. The [receipt](receipt.json)
pins the source baseline, wrapper, profile, frontend, object, toolchain,
simulator, ELFs, and logs. Replay reproduced the object, physical program,
ELFs, and full-output logs byte for byte. This is a fenced serial schedule
for synthetic data; a real TinyLlama head and performance remain separate
qualification tasks. No FPGA was used.

To replay, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to the reference implementation’s MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_llama_vb_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
