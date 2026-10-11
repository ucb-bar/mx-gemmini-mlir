# Two-layer TinyLlama whole-stage Spike diagnostic

This is a **random-weight diagnostic**, not a pretrained TinyLlama inference
result or a quantized PyTorch parity certificate. A receipt-bound model2MLIR
prefill graph compiled through Merlin with 19 source-bound MX calls and seven
unique FP4/FP8 kernels. Nicolas's pinned Gemmini Spike extension ran the
single-stage ELF to `DONE` with a 2 GiB model arena. The raw console is in
[`spike.log`](spike.log).

| Bound item | Digest or count |
| --- | --- |
| model2MLIR `model.mlir` SHA256 | `78980b12f4696c0007e8d079b2d57a44b77821141dfe586d490462565f66bd6b` |
| MX+VPU profile SHA256 | `e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d` |
| Nicolas Spike extension SHA256 | `5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5` |
| Merlin build hash | `85d13340de9e` |
| Routed MX calls | 19 |
| Unique kernels | 7 |
| Model arena | 2 GiB |

The full first-output record covers 32,000 f32 logits. The target output
SHA256 is `756897f97073152ef0cc3eba4ae09be41a2295d688ebf3816dcdaaf52a6b12aa`.
The original unquantized PyTorch `golden.npy` has a different digest and
argmax (21,282 versus target 26,705). FP4/FP8 conversion changes the program;
an independent reference using MX mesh arithmetic is required to judge the
target result.

## Numerical localization

The [host-only Merlin run](host_baseline.json) of the same captured graph,
inputs, and weights agrees with the unquantized PyTorch golden to `1.31e-5`
maximum absolute error and chooses the same top token. The captured graph and
host calling convention therefore passed this two-layer check.

The MX target C encoder and the latest model2MLIR MX operand encoder produced
identical FP4 codes and E8M0 scales for both operand layouts over 128 rows of
the actual layer-0 Q-projection weight (262,144 elements); see the
[comparison record](actual_weight_quant_comparison.json). A 128×64×2048 FP4
kernel produced all 8,192 BF16 results bit for bit against the independent
Radiance mesh model on pinned Spike; the [raw log](multi_tile_spike.log)
records zero mismatches. These checks cover operand conversion and the tested
tiling geometry, not all model shapes or the full graph.

The [standalone Q projection](first_qproj_comparison.json) reused the captured
activation and weight. Across 70 spread-out output samples, its cosine against
model2MLIR's PyTorch BF16 fake-quantization proxy was `0.999982` and maximum
absolute difference was `0.03125`; its [Spike log](first_qproj_spike.log)
contains the output bits. At the full stage output, the
[32,000-logit comparison](fake_quant_comparison.json) fell to cosine `0.618`
and chose a different top token. The fake-quantization proxy does not model
the MX mesh's internal product and reduction arithmetic.

A [complete diagnostic trace](device_trace_spike.log) sampled 24 elements
after each of the 19 MX calls and reached the same final output SHA256 as the
ordinary run. The [per-call comparison](device_trace_comparison.json) records
the samples and reference names. The
first Q/K/V projections stayed close to the fake-quant proxy; the first
attention QK result had sampled cosine `0.999313` and maximum absolute
difference `0.9375`. The sampled first-layer MLP down projection fell to
cosine `0.746`. These samples locate where the proxy diverges; they do not
establish a target bug or a numerical tolerance. An independent full-graph
reference with MX mesh arithmetic remains required.

## Session handoff

The target session linker combined this prefill and its captured decode stage
with Merlin's version-2 session ABI and one byte-verified weight blob. On
pinned Spike, the linked prefill's three output/state digests matched the
separately built prefill, and decode step 0 produced logits, KV-cache state,
and position-state digests before entering step 1. The
[console prefix](session_prefill_decode0_prefix.log) is a handoff smoke test.
That first link exhausted the temporary guest arena at decode step 2 because
the allocator was not reset between recurrent steps. The target session
driver now resets temporary allocations before each invocation; its complete
32-step run is pending. This remains a random-weight diagnostic, not
pretrained inference or numerical qualification.
