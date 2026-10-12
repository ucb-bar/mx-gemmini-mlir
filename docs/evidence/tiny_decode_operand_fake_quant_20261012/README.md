# Full-checkpoint TinyLlama decode operand diagnostic

The [result](result.json) comes from the receipt-bound 22-layer TinyLlama
decode capture and its saved first-step input state. The tool verified the
capture's model, weights, inputs, and golden hashes, reconstructed the PyTorch
fixed-cache decode step, and reproduced its unquantized logits exactly. It
then applied the compiler's selected operand formats to all 200 MX sites:
FP4 to 155 Linear contractions and FP8 to 44 attention matmuls plus the
rotary K=1 BMM. The latter is padded to one 32-element MX block before
quantization, as in the target's physical path.

| Output | First logit | Top token |
| --- | ---: | ---: |
| Captured unquantized PyTorch | -9.377923965454102 | 29871 |
| PyTorch FP4/FP8 operand proxy | -11.0 | 319 |
| [Compiled MX+VPU decode on Spike](../tiny_vpu_decode_spike_20261012/README.md) | -3.6875 | 27305 |

The proxy's maximum absolute change from the unquantized logits is
8.622732162475586; its mean absolute change is 1.4643105268478394. These
figures show that the requested operand precision materially changes this
captured step. The proxy uses PyTorch BF16 matmuls; it does not reproduce the
MX mesh's product and accumulation arithmetic or the VPU's intermediate BF16
boundaries and approximations. Its difference from Spike therefore does not,
by itself, identify a compiler defect or qualify numerical correctness.
This is a one-step decode on captured input state, not a recurrent inference
session or a model-quality evaluation.

For a local capture and model2MLIR checkout, rerun
[`tools/diagnose_tinyllama_operand_quant.py`](../../../tools/diagnose_tinyllama_operand_quant.py)
with:

```sh
python -m tools.diagnose_tinyllama_operand_quant \
  --stage /path/to/tiny_llama_session/stages/decode \
  --model2mlir-root /path/to/model2mlir \
  --out-dir /path/to/new/diagnostic-output \
  --layers 22 --capacity 160 --expected-fp4 155 \
  --expected-fp8 44 --expected-fp8-bmm 1 --stage-kind decode
```

No checkpoint, token sequence, or logit vector is published here. The result
records the artifact identities and proxy output digest without local paths.
