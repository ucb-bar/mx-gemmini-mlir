# Typed 2,048-key streamed flash attention on Spike

The [source baseline](../nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb/README.md)
uses Nicolas's unmodified `attn_flash_llama.c` and synthetic data from his
generator at 64 queries, 2,048 keys, dimension 64, and `BK=256`. The source
selects K×2 and V×3 scratchpad rings. Its model2MLIR `e9ded36` capture
provides the QK→scale→softmax→PV frontend graph.

The [typed graph](flash_llama_stream.mlir) binds that frontend and source to
35 transfer phases, 16 managed matmuls, 83 VPU operations, and eight
probability requants. It places K in scratchpad rows 4096/5120, V in
256/1280/2304, probability in 6144/7168, scores in 8192/12288,
statistics at 3328, and output at 14848. K/V blocks load in each serial
iteration before their QK and PV consumers. The public object compiler emits
[484 MX commands and 50 fences](object/physical_program.json) into a
[data-free RV64 object](object/mx_issue.o), with no allocated data section.

The [source](source_spike_1.log) and [compiler](generated_spike_1.log) drivers
each ran twice on Nicolas's pinned Spike extension. **All 4,096 BF16 output
words match bit for bit** across the four runs, with hash `dc33d078bc537b19`.
The [receipt](receipt.json) records the source baseline, frontend, profile,
object, ELF, toolchain, simulator, and log digests. A replay from committed
compiler code reproduced the object and both ELFs byte for byte.

This is a fenced serial compiler schedule. It proves functional behavior on
Spike for this source shape; pipelined compiler scheduling, performance
parity, a real TinyLlama captured head, and causal masking need separate
qualification. No FPGA was used.

To reproduce, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_llama_stream_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
