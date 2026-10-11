# Typed causal streamed flash attention on Spike

Nicolas's unmodified `attn_flash_llama_causal.c` selects 64 queries over
2,048 keys, dimension 64, `BK=256`, and a causal mask on the last 64 keys.
The [data header](attn_vpu_llama_causal.h.gz) was generated with his
`gen_attn_vpu.py --sq 64 --sk 2048 --d 64 --seed 0 --tag
_llama_causal_probe --no-s-golden --causal`, then copied under the canonical
header filename. It contains **synthetic operands**, not a captured TinyLlama
head. The source oracle's output hash is `23d84ffca3505276`.

The [PyTorch graph](attention.model2mlir.mlir) was captured by model2MLIR
`e9ded36` with the FX importer and no opaque operations. Its SSA chain is
QK → multiply by 0.125 → add a runtime causal mask → softmax → PV. The
[capture tool](../../../tools/capture_nicolas_flash_llama_causal.py) checks
the exact frontend digest. The source data header fixes the mask values;
the model2MLIR graph fixes the operation and tensor shapes.

The [typed MX graph](flash_llama_causal_stream.mlir) binds that frontend to
the canonical source and the selected MX+VPU profile. Its final runtime
argument is the 8,192-byte BF16 mask. It transfers the mask to scratchpad
row 10752, then issues 64 row-wise VPU adds against the final score block's
last 64 columns. The other K×2 and V×3 streamed buffers, gated scales,
requants, and output readout use the existing flash lowerer. The public
compiler emits [581 MX commands and 50 fences](object/physical_program.json)
into a [data-free RV64 object](object/mx_issue.o).

The [source](source_spike_1.log) and [compiler](generated_spike_1.log)
drivers each ran twice on the pinned MX Spike extension. **All 4,096 BF16
output words match bit for bit** across all four runs. The [receipt](receipt.json)
pins source, frontend, generator, profile, toolchain, simulator, object,
ELFs, and logs. Replay reproduced the object, physical program, ELFs, and
logs byte for byte. This qualifies functional parity for the selected
synthetic causal shape and fenced serial schedule. It does not qualify a
real TinyLlama capture or performance. No FPGA was used.

To replay, set `NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_llama_causal_typed \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```
