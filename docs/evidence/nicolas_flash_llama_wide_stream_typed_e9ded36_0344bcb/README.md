# Two-V-buffer streamed flash attention on Spike

the reference implementation’s unmodified `attn_flash_llama_wide.c` wrapper selects the same
64×2,048×64 synthetic operands, uniform `BK=256`, and model2MLIR frontend
as the [three-V-buffer source baseline](../nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb/README.md).
It sets `ATTN_V3=0`: K remains double buffered and V uses two scratchpad
buffers. The source output hash remains `dc33d078bc537b19`.

The [typed graph](flash_llama_wide_stream.mlir) binds the wrapper policy,
places V in scratchpad rows 256 and 1280, and places the softmax statistics
at row 2304. It lowers the same QK, VPU softmax, requant, PV, and output
schedule through the public compiler. The [data-free RV64 object](object/mx_issue.o)
contains [484 MX commands and 50 fences](object/physical_program.json).

The [source](source_spike_1.log) and [compiler](generated_spike_1.log)
drivers each ran twice on the pinned MX Spike extension. **All 4,096 BF16
output words match bit for bit** across the four runs. The [receipt](receipt.json)
pins the source baseline, wrapper, frontend, profile, object, toolchain,
simulator, ELFs, and logs. Replay reproduced the object, physical program,
ELFs, and full-output logs byte for byte. This qualifies the two-buffer
functional mode for a fenced serial schedule and synthetic operands. It
does not qualify performance or a real TinyLlama head. No FPGA was used.

To replay, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to the reference implementation’s MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_llama_wide_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
