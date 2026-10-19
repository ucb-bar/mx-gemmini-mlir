# Four model-derived FP4 projections on pinned MX Spike

This archive contains one **complete layer-0 `q_proj` matrix multiplication**
per reduced model2MLIR graph. Each program uses actual operands tapped from
one deterministic random-weight model forward, then BF16 operand conversion,
MXFP4 E2M1 quantization, E8M0 scales per 32 elements, direct nibble packing,
the pinned `MxE4M3Fp4VpuGemminiRocketConfig`, and compiler-generated MX
commands. The independent Radiance `mx_golden.cpp` result matched every BF16
value from the reference implementation’s pinned MX Spike model. The source profile identifies the
MX+VPU RTL, although these programs use the matrix engine only.

| Model | Source matmul M×N×K | Compared BF16 values, including padded rows | Real model values | Mean absolute error vs PyTorch f32 |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 8×2048×2048 | 65,536 | 16,384 | 0.318 |
| DeepSeek-R1-Distill-Qwen-1.5B | 4×1536×1536 | 49,152 | 6,144 | 0.280 |
| Gemma 2 2B | 4×2048×2304 | 65,536 | 8,192 | 0.344 |
| SmolVLA | 16×768×768 | 24,576 | 12,288 | 0.200 |

The FP4 tile requires 32 rows, so short activation batches are zero-padded;
the real output counts exclude those rows. Qwen and SmolVLA's `addmm` bias-add
epilogues are outside these programs. The archive therefore proves one
full-width matmul at each model site, not an entire attention block, model,
prefill/decode session, or action sequence. The models are reduced and randomly
initialized. Gemma uses the installed default config because the gated
checkpoint config was unavailable offline. These absolute errors are
diagnostics; acceptable model quality still needs calibration and end-to-end
inference tests with the intended checkpoints.

Each model directory contains a graph/source receipt, the exact packed operand
bundle, typed and payload-bound MLIR, compressed physical command program,
data-free object receipt, Spike receipt, and output log. The ELF digest in the
receipt binds the executable that ran, while the compiled ELF can be rebuilt
from the archived payload and pinned toolchain.

Run the archive checks with:

```sh
python -m pytest -q tests/test_four_model_fp4_projection_evidence.py
```

`tools.prepare_tinyllama_projection_inputs` regenerates TinyLlama's floating
operands; `tools.prepare_model_projection_inputs` does the same for the other
three. Recompile a site with `tools.qualify_model_projection --precision fp4`,
its archived `.mlir.gz` graph, saved `activation.npy` and `weight.npy`, the
pinned MXQuant and Radiance arithmetic sources, the MX+VPU profile, and
`--run-spike`. The `.npy` arrays stay in the task's temporary directory; this
archive preserves the packed arrays and their source digests.
