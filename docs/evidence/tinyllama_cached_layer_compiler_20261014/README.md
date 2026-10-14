# TinyLlama cached layer: compiler section probe

This probe starts with the real TinyLlama checkpoint and the 128 token IDs
captured by model2MLIR for the full prefill session. The
[`tiny_llama_layer` loader](https://github.com/ucb-bar/model2MLIR/tree/cx/mx-vpu-four-model-e2e/workloads/tiny_llama_layer)
extracts layer 5, a 64 token KV cache, and the following 64 query tokens.
It checks the cached PyTorch layer against the full checkpoint. The resulting
model2MLIR bundle has zero opaque operations, nine externalized checkpoint
weight tensors, and a 64×2048 output. The nine weight tensors match the
[Nicolas source capture](../nicolas_tinyllama_layer5_captured128_20261014/README.md)
byte for byte.

The selected MX profile routes all nine contractions to MX. It routes one
softmax, four adds, 16 elementwise operations, seven unary operations, and
two reductions to VPU kernels. The other operations remain in the Merlin
host graph. The FP8 recipe overrides seven default FP4 projections; both
attention contractions were already FP8.

| Compiler stage recipe | Spike `METRIC cycles` | Row argmax vs PyTorch |
| --- | ---: | ---: |
| Default FP4 projections, FP8 attention, cached weights | 3,669,741,534 | 44/64 |
| FP8 throughout, cached weights | 3,539,927,144 | 59/64 |
| FP8 throughout, weights packed at build time | 268,345,341 | 59/64 |

The build-time packed FP8 output is 7.60% relative Frobenius error from
the same PyTorch layer. The independently written Nicolas FP8 source layer
reported 7.10% error. Its timed kernel took 582,204 Spike `rdcycle`
counts. **These cycle values are not a speedup comparison:** the compiler
counter covers the standalone stage with its representation conversions;
Nicolas starts after its binary operands are already packed and stops when
`h_out` reaches DRAM. The compiler uses f32 graph boundaries and separately
materialized MX/VPU calls. The source keeps packed intermediates resident
and fuses the attention schedule. The outputs also differ numerically, so
equal-output performance parity is not established.

An instrumented run of the build-time packed FP8 stage attributed 99.86%
of its 268,359,570 cycles: 154,759,889 to MX calls, 106,792,209 to VPU
calls, and 6,439,869 to other host operations. This locates the remaining
cost in the per-operation accelerator path, including packing, conversions,
and transfers. It does not isolate hardware busy time. The generated
`mx_dense_kernel_3.c`, for example, calls `mx_pack_activation_f32`, issues
one MX kernel, then converts its BF16 output back to f32. The corresponding
source program carries packed data between phases. The instrumented build
is diagnostic; use the uninstrumented row above for stage timing.

[`summary.json`](summary.json) binds the captured MLIR, selected profile,
Spike extension, run receipts, and output digests. [`output_metrics.json`](output_metrics.json)
contains the full-output checks against the bundle golden and Nicolas's
reference tensor. [`stage_profile.json`](stage_profile.json) contains the
source-region operation attribution. The 169 MB checkpoint weights and
ELFs stay in the task artifacts; the [model2MLIR loader](https://github.com/ucb-bar/model2MLIR/tree/cx/mx-vpu-four-model-e2e/workloads/tiny_llama_layer)
and [full-model guide](../../compile_e2e_models.md) describe how to recapture
and compile them. The TinyLlama source section is reproduced by the script
in its [source evidence directory](../nicolas_tinyllama_layer5_captured128_20261014/reproduce.sh).
