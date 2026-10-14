# Matched two-layer TinyLlama VPU class controls

These Spike runs use the first two decoder layers of the **same pretrained
TinyLlama checkpoint** as the full-model capture, followed by the final norm
and language-model head. They use the same 64 token IDs, all-FP8 MX precision
overrides, 20 selected MX contractions, generated checkpoint-weight object,
and MX catalog object. Only the selected VPU postprocessing class changes.
The reference is the complete final-position vocabulary row captured from
PyTorch, with next-token argmax 5056.

| VPU classes enabled | VPU sites | Stage cycles | Cosine to PyTorch | Argmax |
| --- | ---: | ---: | ---: | ---: |
| None; host control | 0 | 432,913,793 | 0.9911 | 5056 |
| Add | 8 | 455,307,913 | 0.9912 | 5056 |
| Unary | 16 | 402,423,585 | 0.9918 | 5056 |
| Reduce | 5 | 443,572,466 | 0.9916 | 5056 |
| Add + unary + reduce | 29 | 432,078,902 | 0.9915 | 5056 |
| Softmax | 2 | 409,574,642 | 0.9140 | 1926 |
| Elementwise | 38 | 546,515,911 | 0.9171 | 1926 |
| All except softmax | 67 | 533,522,648 | 0.9144 | 1926 |
| All five classes | 69 | 521,264,750 | 0.9152 | 1926 |

Softmax and elementwise each reproduce most of the two-layer logit drift by
themselves. The controls isolate the VPU postprocessing route from the MX
contractions and checkpoint weights. This does not yet distinguish a command
bug from BF16 precision loss; it also does not establish model accuracy from
one captured input. The independent FP8 source generator has cosine 0.9968
to this PyTorch reference.

The [summary](summary.json) records input, profile, precision, weight, catalog,
Spike-log, and output hashes. Each row's complete output is in its compressed
`<row>.log.gz` file. The [reference logits](pytorch_last_logits.npy) and
[input IDs](inputs.npz) are archived. Run `python verify.py` to recheck every
output digest, cosine, argmax, and cycle count with NumPy.

To reproduce a row, build the captured prefill stage with
`tools.build_merlin_stage`, the pinned MX+VPU profile, the all-FP8 override
map, `--prepack-checkpoint-weights`, and the pinned Spike extension. Use
`--diagnostic-host-vpu` for the host control or
`--diagnostic-vpu-classes softmax,add,elementwise,unary,reduce` with the
desired subset for a VPU control. The [model compilation guide](../../compile_e2e_models.md)
describes the stage build and its numeric-qualification status.
