# Matched two-layer TinyLlama VPU class controls

These Spike runs use the first two decoder layers of the **same pretrained
TinyLlama checkpoint** as the full-model capture, followed by the final norm
and language-model head. They use the same 64 token IDs, all-FP8 MX precision
overrides, 20 selected MX contractions, generated checkpoint-weight object,
and MX catalog object. Only the selected VPU postprocessing class changes.
The reference is the complete final-position vocabulary row captured from
PyTorch, with next-token argmax 5056.
Capture with `M2M_LLAMA_CHECKPOINT_LAYERS=2` on the model2MLIR
`cx/mx-vpu-four-model-e2e` branch. This loads the pretrained checkpoint and
keeps its first two layers; `M2M_LLAMA_LAYERS=2` creates a different random
model and is not this control.

| VPU classes enabled | VPU sites | Stage cycles | Cosine to PyTorch | Argmax |
| --- | ---: | ---: | ---: | ---: |
| None; host control | 0 | 432,913,793 | 0.9911 | 5056 |
| Residual add | 8 | 455,307,913 | 0.9912 | 5056 |
| Unary | 16 | 402,423,585 | 0.9918 | 5056 |
| Reduce | 5 | 443,572,466 | 0.9916 | 5056 |
| Residual add + unary + reduce | 29 | 432,078,902 | 0.9915 | 5056 |
| Softmax | 2 | 409,574,642 | 0.9140 | 1926 |
| Elementwise | 38 | 546,515,911 | 0.9171 | 1926 |
| Elementwise divide only | 5 | 436,342,869 | 0.9915 | 5056 |
| Elementwise multiply only | 26 | 544,293,328 | 0.9921 | 5056 |
| Elementwise add only | 7 | 433,205,874 | 0.9149 | 1926 |
| Attention-shaped elementwise adds | 2 | 433,182,968 | 0.9155 | 1926 |
| Layer-0 attention add only | 1 | 433,524,962 | 0.9869 | 5056 |
| Layer-1 attention add only | 1 | 432,606,919 | 0.9612 | 5056 |
| All except softmax and two attention adds | 65 | 534,655,868 | 0.9915 | 5056 |
| All except softmax | 67 | 533,522,648 | 0.9144 | 1926 |
| All five classes | 69 | 521,264,750 | 0.9152 | 1926 |

Softmax and elementwise each reproduce most of the two-layer logit drift by
themselves. Within elementwise, the seven adds reproduce the drift, and the
two attention-shaped adds are sufficient; divide-only and multiply-only do
not. Layer 0 alone has cosine 0.9869 and layer 1 alone has cosine 0.9612;
both preserve the token. Together they compound to 0.9155 and change it.
Keeping 65 VPU sites while leaving softmax and those two attention additions
on the host retains 0.9915 cosine and the reference token. This is a
two-layer control; it does not yet qualify a 22-layer policy.
The controls isolate VPU postprocessing
from the MX contractions and checkpoint weights. This does not yet distinguish
a command bug from BF16 precision loss; it also does not establish model accuracy from
one captured input. The independent FP8 source generator has cosine 0.9968
to this PyTorch reference.

A [standalone attention-shaped add check](attention_add_bf16_qualifier.json)
on the same Spike extension used a `[1,32,64,65]` tensor, broadcast mask,
and periodic negative-infinity mask values. It had zero mismatches against
an independent BF16 add loop. This supports the command implementation for
that fixture; it does not establish that rounding FP32 attention scores to
BF16 preserves the model's logits. The [Spike log](attention_add_bf16_qualifier.log)
and generated [driver](attention_add_driver.c) and
[MX issuer](attention_add_issue.c) are archived with the receipt.

The [summary](summary.json) records input, profile, precision, weight, catalog,
Spike-log, and output hashes. Each row's complete output is in its compressed
`<row>.log.gz` file. The [reference logits](pytorch_last_logits.npy) and
[input IDs](inputs.npz) are archived, along with the
[all-FP8 override map](all_fp8_overrides.json). Run `python verify.py` to
recheck every output digest, cosine, argmax, and cycle count with NumPy.

To reproduce a row, build the captured prefill stage with
`tools.build_merlin_stage`, the pinned MX+VPU profile, the all-FP8 override
map, `--prepack-checkpoint-weights`, and the pinned Spike extension. Use
`--diagnostic-host-vpu` for the host control or
`--diagnostic-vpu-classes softmax,add,elementwise,unary,reduce` with the
desired subset for a VPU control. Add `--diagnostic-elementwise-ops add` to
select one elementwise operation type, or use
`--diagnostic-elementwise-regions 'add_5#add0,add_12#add0'` to select exact
source regions. The [model compilation guide](../../compile_e2e_models.md)
describes the stage build and its numeric-qualification status.
