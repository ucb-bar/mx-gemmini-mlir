# Full-checkpoint TinyLlama operand fake-quant diagnostic

The [result](result.json) comes from the receipt-bound, 22-layer TinyLlama
prefill capture with its 4,400,217,104-byte checkpoint weight artifact. The
diagnostic validated the capture's model, weights, inputs, and golden hashes,
reconstructed its fixed-cache PyTorch stage, and reproduced the saved
unquantized logits exactly. It then applied the proposed MX placement: FP4
to all 155 Linear contractions and FP8 to all 44 attention QK/PV matmuls.

| Measurement | Result |
|---|---:|
| Unquantized top token | 279 |
| FP4/FP8 proxy top token | 29871 |
| Maximum absolute logit change | 11.758407592773438 |
| Mean absolute logit change | 2.067028760910034 |

The tool is [`tools/diagnose_tinyllama_operand_quant.py`](../../../tools/diagnose_tinyllama_operand_quant.py).
For a local capture and model2MLIR checkout, rerun it with:

```sh
python -m tools.diagnose_tinyllama_operand_quant \
  --stage /path/to/tiny_llama_session/stages/prefill \
  --model2mlir-root /path/to/model2mlir \
  --out-dir /path/to/new/diagnostic-output \
  --layers 22 --capacity 160 --expected-fp4 155 --expected-fp8 44
```

This uses the current model2MLIR `mxq` operand quantizer and PyTorch BF16
matrix arithmetic. It does not model the MX mesh's product floor, accumulation
schedule, or the VPU softmax boundary. The logits are a placement and model
quality diagnostic, **not an accelerator correctness oracle**. The capture
artifacts were generated before the diagnostic's recorded model2MLIR revision;
the diagnostic binds them by their validated artifact hashes. No checkpoint,
token sequence, or logit vector is published in this evidence folder.
