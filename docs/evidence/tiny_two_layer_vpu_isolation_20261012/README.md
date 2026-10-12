# Two-layer TinyLlama decode VPU isolation on Nicolas's Spike

Two RV64 programs compiled the same receipt-bound, random-weight TinyLlama
decode capture under the same MX+VPU profile. Both route the same 20 source
contractions to MX, including the rotary K=1 BMM. Their matrix catalog JSON,
matrix object, device shim, and pre-VPU prepared MLIR are byte-identical. The
diagnostic build leaves VPU-eligible arithmetic on Merlin's host route; the
target build lowers two softmax, eight add, 36 other binary, 16 unary, and seven
row-reduction source regions to the VPU.

Both programs completed on Nicolas's pinned Spike extension with guest `DONE`,
zero memref rank mismatches, and complete logits, KV-cache, and position-state
digests. Their results differ:

| Route | First logit | Top token | Logits digest | KV-cache digest |
| --- | ---: | ---: | --- | --- |
| MX plus host arithmetic control | 1.6640625 | 26686 | `ca5335dfca4c02e4…` | `6cc4f8e8273ac532…` |
| MX plus VPU | 0.59375 | 28954 | `0c2d4fc60679fab3…` | `4acda1be1e10020f…` |

The position-state digests match. [The evidence index](index.json) binds the
complete digests, ELF and run receipt hashes, profile, capture, extension, and
byte-identical pre-VPU artifacts without local file paths. This comparison
isolates the numerical effect of the VPU rewrite for one captured decode step.
It does not establish which output is numerically correct: the VPU uses BF16
boundaries and its own approximations, and there is no independent full-graph
quantized reference yet. The model weights are random; this is a compiler
diagnostic, not pretrained inference or an accuracy measurement.

The control was built with `tools.build_merlin_stage --diagnostic-host-vpu` and
run with `tools.run_linked_merlin_stage`. The session linker rejects diagnostic
builds so this route cannot silently become the production session image.
