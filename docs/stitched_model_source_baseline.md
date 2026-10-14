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
| SmolVLA flow: 15 heads, 50 queries, 163 keys, Boolean key mask, 148 selected MX bindings | [`attn_vpu_fa.c`][npu-attn] fixes dense 64×256×128 attention | Pad and mask the model-derived heads, preserve the 50×163 logical work, and stitch attention with the remaining flow operators. |

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

The [equal-work dense attention kernel](evidence/attn_vpu_equal_work_20261013/README.md)
already compares one fused MX+VPU source shape with compiler-issued commands.
It does not establish a whole-model stitched baseline for the captured shapes.

An [independent first-two-layer TinyLlama source
control](evidence/nicolas_tinyllama_full_source_vpu_20261014/README.md) now
uses Nicolas's full-model MX schedule with his VPU residual-add pattern. It
runs the final norm and head as well. All 20 source and model2MLIR FP32
checkpoint tensors and all 64 input token IDs match exactly. The source's
timed full stage takes 366,920,372 Spike cycles versus 455,307,913 for the
compiler's residual-add control; their final logits have 0.9958 cosine and
the same top token for this input. The source diagnostic checks pass and its
timed output is bit identical to the diagnostic output. This is two layers,
not the 22-layer or 33-step TinyLlama model, and has no accepted model-level
quality or speedup claim.

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
