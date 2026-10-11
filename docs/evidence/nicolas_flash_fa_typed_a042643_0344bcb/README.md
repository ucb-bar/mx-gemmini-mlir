# Typed default flash attention shape on Spike

This archive reproduces Nicolas's default `attn_flash.c` kernel with one
compiler-generated RV64 RoCC object. The kernel has `Sq=64`, `Sk=256`,
`d=128`, and four `Bk=64` key blocks. The clean model2MLIR `a042643`
capture binds the QK→scale→softmax→PV dataflow. The selected RTL is
`gemmini-mx-cleanup` `266c593` with VPU. This shape uses an explicit
`1/√128` BF16 VPU multiply on every score block; the smaller `d=64`
variant folds its scale into Q's E8M0 bytes.

The [typed graph](flash_fa.mlir) lowers through the shared
[physical command IR](physical_program.json) to **236 MX commands and 21
fences**. It includes Q/K/V DMA, managed scale-half QK/PV matmuls, online
softmax, resident FP8 requantization, output updates, and BF16 readout.
The [RV64 object](mx_issue.o) contains no operands, expected values, or
data section. Its [manifest](object_manifest.json) defines ten runtime
buffers. The compiler uses a fenced serial schedule for functional parity;
Nicolas's source runs a pipelined schedule. Performance equivalence is not
claimed.

The [driver](driver.c) runs Nicolas's original source for the reference
path and calls only the compiler-generated `mx_issue` for the generated
path. Both ran twice on the pinned Spike extension. Every one of the
**8,192 BF16 output values** matches bit for bit across the four
[source](source_spike_1.log) and [generated](generated_spike_1.log) logs.
The common output hash is `fba97c66241b47bf`; the source's checker
reports `attn_flash PASSED`. The [receipt](receipt.json) pins the source,
frontend, toolchain, native verifier, object, ELFs, and logs. A fresh replay
reproduced those files byte for byte.

To reproduce from a clean checkout, build `mx-gemmini-opt`, then set
`NPU_ROOT` to `npu-exploration` `0344bcb`, `RTL_ROOT` to
`gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX Spike extension, and `OUT_DIR` to a fresh
directory:

```sh
python -m tools.qualify_nicolas_flash_mid_typed --variant fa \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OUT_DIR"
```

To build only the same linkable object through the public compiler, use a
different fresh output directory. The default variant uses `attn_flash.c`
as both its source core and wrapper:

```sh
python -m tools.compile_object \
  --mlir docs/evidence/nicolas_flash_fa_typed_a042643_0344bcb/flash_fa.mlir \
  --precursor-mlir docs/evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/attention.model2mlir.mlir \
  --binding-json docs/evidence/nicolas_flash_fa_typed_a042643_0344bcb/binding.json \
  --source-core "$NPU_ROOT/baremetal/mxgemmini/src/attn_flash.c" \
  --source-wrapper "$NPU_ROOT/baremetal/mxgemmini/src/attn_flash.c" \
  --source-header docs/evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/attn_vpu_fa.h.gz \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt "$MX_GEMMINI_OPT" --out-dir "$OBJECT_OUT_DIR"
```

This qualifies the E4M3, noncausal, resident K/V source shape on the pinned
MX+VPU configuration. FP4, causal masks, fused VPU variants, streamed K/V,
and pipelined scheduling remain separate work.
