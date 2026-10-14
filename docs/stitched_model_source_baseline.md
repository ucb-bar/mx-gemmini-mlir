# Matched MX+VPU source baseline for captured models

The [full TinyLlama](evidence/nicolas_matrix_substitution_tinyllama_full_20261014/README.md)
and [full SmolVLA](evidence/nicolas_matrix_substitution_smolvla_full_20261013/README.md)
Spike comparisons replace compiler MX matrix issuers with Nicolas's native
issuers. They keep the captured host graph and VPU objects fixed. They measure
all session steps and check every output digest, but they are not a stitched
handwritten MX+VPU implementation of either complete graph.

Nicolas's pinned [source roster][npu-source] provides fused attention, MLP,
and TinyLlama layer kernels. Its interfaces and data policies differ from the
captured sessions:

| Captured work | Available source kernel | Required adaptation |
| --- | --- | --- |
| TinyLlama prefill: 128 tokens, 22 layers, 155 FP4 and 45 FP8 MX bindings per stage | [`llama_layer_e2e.c`][npu-layer] uses a 64-token chunk after a 2048-token cache; [`llama_mlp_e2e_fp4.c`][npu-mlp] covers an FP4 MLP only | Preserve the captured prompt, checkpoint weights, mixed precision, 128-token causal mask, and full layer stack in one stage ABI. |
| TinyLlama decode: one query token repeated for 32 steps, 155 FP4 and 45 FP8 MX bindings | The same layer source plans work in multiples of 16 tokens; [`llama_model.c`][npu-model] uses FP8 mesh work with scalar host stages | Add a padded one-token MX+VPU path and stateful KV handoff with the captured numeric policy. |
| SmolVLA prefix: 243 selected MX bindings | No SmolVLA whole-model source program in the pinned MX application roster | Match the encoder, mask and cache outputs, and FP4/FP8 policy at the stage ABI. |
| SmolVLA flow: 15 heads, 50 queries, 163 keys, head width 64, Boolean key mask, 148 selected MX bindings | [`attn_vpu_fa.c`][npu-attn] fixes dense 64×256×128 attention; [`attn_flash.c`][npu-flash] has a VPU additive causal mask | Pad each head to 64 queries and 192 keys, load the model's arbitrary Boolean mask as BF16 additive values before row maximum, suppress padded rows/keys, and stitch attention with the remaining flow operators. |

A source baseline is comparable on slide 3 when it uses the **same captured
inputs, checkpoint and profile**, runs the same 33 TinyLlama or 12 SmolVLA
steps on the same `e4m3_vpu` Spike timing model, and reproduces the declared
outputs. The current full-session comparator verifies the program schedule,
weights, Spike settings, output digests, and changed stage objects for issuer
swaps. A fused replacement may change the stage object set; its comparison
receipt must name those replacements while preserving the other checks.
If a fused source implementation deliberately changes
rounding, its output error and model quality need separate qualification; a
cycle total alone is not an equal-output result.

The SmolVLA mask is an arbitrary `1×1×50×163` Boolean tensor in the captured
flow MLIR. The [pinned attention cutpoint](evidence/smolvla_attention_cutpoint_20261014/README.md)
now archives actual Q/K/V operands, mask, and softmax probabilities from the
first denoising step. Recomputed masked softmax matches the captured
probabilities below `2e-9` RMSE. The unmodified flash source applies a
causal mask only to its last key block. A small source patch now applies
SmolVLA's Boolean mask through `VPU_ADD` on **every** block. All 15 heads
of the first flow step pass separately on Spike at padded 64×192×64, with
22,447 pipelined modeled cycles per separately timed head and 2.977–6.356%
relative output error versus each captured FP32 attention result. Other
flow steps and the rest of the model still need a stitched source run before
this becomes a full-model cycle control.

The [equal-work dense attention kernel](evidence/attn_vpu_equal_work_20261013/README.md)
already compares one fused MX+VPU source shape with compiler-issued commands.
It does not establish a whole-model stitched baseline for the captured shapes.

The separate [Radiance kernel comparison](evidence/radiance_kernel_comparison_20261014/README.md)
checks the compiler against current FP8/FP4 GEMM, batched GEMV, and GQA MX
source controls. The read-once GEMMs match all BF16 source outputs and the
source's one-read weight traffic; the four-pass FP8 control transfers four
times as many weight bytes. Radiance's source kernels launch MX through
Muon, so their cycles cannot be compared with the standalone Rocket Spike
model cycles. Its existing connected `model_chain` programs use scalar Muon
stages and do not supply a full MX+VPU model baseline.

An [independent first-two-layer TinyLlama source
control](evidence/nicolas_tinyllama_full_source_vpu_20261014/README.md) now
uses Nicolas's full-model MX schedule with his VPU residual-add pattern. It
runs the final norm and head as well. All 20 source and model2MLIR FP32
checkpoint tensors and all 64 input token IDs match exactly. The source's
timed full stage takes 366,920,372 functional Spike cycles versus 455,307,913
for the compiler's residual-add control; their final logits have 0.9958
cosine and the same top token for this input. The source diagnostic checks
pass, and its timed output is bit identical to the diagnostic output. This is
two layers, not the 22-layer or 33-step TinyLlama model. It carries no
model-level quality or speedup claim.
Those two counts use the extension's default `func` mode; the full-depth
performance experiment runs both programs with Nicolas's `e4m3_vpu` timing
model enabled.

The [independent full-depth TinyLlama prefill
control](evidence/nicolas_tinyllama_full_source_vpu_20261014/README.md)
now runs all 22 layers and the head on 64 captured tokens. The source
and compiler captures have identical token IDs and all 200 FP32 checkpoint
tensors. With `GEMMINI_MODE=both` and the `e4m3_vpu` model, the source
reports **4,328,997,732** stage cycles and the compiler **5,072,967,821**.
Their final-logit cosine is **0.9816** and both select token **1230**. The
source VPU variant is bit-identical to its earlier scalar-host source result.
This is a one-pass prefill diagnostic: the source calls the VPU for 44
residual adds, while the compiler lowers 88 add regions and uses different
host math. The raw cycle difference is not a full 33-step model speedup.

The [independent 128-token prefill](evidence/nicolas_tinyllama_prefill128_source_20261014/README.md)
uses the same 200 FP32 checkpoint tensors and 128 input IDs on both sides.
Nicolas's source takes **9,611,186,417** modeled cycles and the all-FP8
compiler prefill takes **11,442,665,332**. Their final logits have cosine
**0.9454** and different top tokens. The source reaches 0.9932 cosine to
captured FP32 logits; the compiler reaches 0.9432. Different host/VPU work,
128 versus 160 attention key slots, and timing boundaries still prohibit a
speedup ratio. This is one prefill stage, not the 33-step session or a
mixed FP4/FP8 source baseline.

The [captured TinyLlama layer-5 source run](evidence/nicolas_tinyllama_layer5_captured128_20261014/README.md)
uses the captured token IDs and byte-identical layer weights. The
[compiler section probe](evidence/tinyllama_cached_layer_compiler_20261014/README.md)
now captures the same 64 cached plus 64 query tokens from PyTorch and
compiles all nine layer contractions with FP8. Both sections run on the
same Spike extension, but their inputs, intermediate representations,
numeric outputs, and timed work boundaries still differ. Their reported
cycle values are therefore not a paired speedup or a stitched model result.

[npu-source]: https://github.com/ucb-bar/npu-exploration/tree/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src
[npu-layer]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/llama_layer_e2e.c
[npu-mlp]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/llama_mlp_e2e_fp4.c
[npu-model]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/llama_model.c
[npu-attn]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/attn_vpu_fa.c
[npu-flash]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/attn_flash.c
