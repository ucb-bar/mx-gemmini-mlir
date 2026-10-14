# Independent TinyLlama source smoke test

Nicolas's pinned `llama_model_m64l2.c` runs two stacked TinyLlama layers on
Spike. This reproduction uses the first 64 token IDs from the model2MLIR
TinyLlama prefill capture and the 65th ID as the next-token label. The source
generator reads the same Hugging Face checkpoint and emits FP8 weights,
operands, and stage goldens. The two layers consume each other's device
output; each mesh, seam, and host stage passed its generated bit-exact check.

| Source timing window | Spike cycles |
| --- | ---: |
| MX mesh sections | 6,199,178 |
| Scalar host sections | 347,476,561 |

The host sections account for 98.25% of these two counters. The final hidden
state differs from the PyTorch capture by 13.28% relative Frobenius norm after
two quantized layers. This is a source-only accuracy measurement, not a
compiler error measurement. The source C times these sections separately, so
their sum is not a whole-ELF cycle count.

**Comparison boundary:** this is a 64-token, two-layer FP8 source smoke test
with scalar host normalization, RoPE, softmax, SiLU, and residuals. The
compiler's full TinyLlama session is a 128-token prefill plus 32 decode steps
through 22 layers, uses mixed FP4/FP8, and lowers selected scalar work to the
VPU. This run cannot be divided into the compiler's cycle count to claim a
speedup. Nicolas's `llama_layer_e2e.c` exercises the optimized MX+VPU path for
one layer; his `llama_model_m64l2.c` exercises stacked depth with host scalar
work. A matched full MX+VPU source model remains to be measured.

The [receipt](receipt.json) pins the source, RTL, quantizer, input, ELF, Spike
extension, and output hashes. `captured_ids.patch` adds `--input-ids` to the
pinned source capture. `limit_workers.patch` caps golden generation at eight
processes. The [reproduction script](reproduce.sh) writes captures, the data
blob, ELF, and log only under `WORK_DIR`. It requires a local cached TinyLlama
checkpoint, the pinned source checkouts, the model2MLIR prefill `inputs.npz`,
and the same Spike extension.
