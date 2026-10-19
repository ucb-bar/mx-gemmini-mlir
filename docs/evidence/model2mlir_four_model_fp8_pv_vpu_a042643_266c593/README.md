# Four model-derived FP8 attention PV heads on pinned MX Spike

This archive executes **one head from the first PV batched matrix
multiplication** in each reduced model2MLIR graph. The probability and value
operands were tapped from a seeded forward pass of each randomly initialized
model's decomposed PyTorch export graph. Each payload names its actual
`linalg.generic` region, source shape, and full graph SHA-256. SmolVLA's site
belongs to the first vision encoder attention layer.

The models' short sequence lengths give PV K dimensions of 8, 4, 4, and 16.
The compiler pads each K dimension with zero codes to a 32-element E8M0 scale
group, while preserving the original K in the source contract. Both operands
use MXFP8 E4M3, with BF16 operand conversion and direct packing. The selected
`MxE4M3Fp4VpuGemminiRocketConfig` includes the VPU, although these heads use
only the MX matrix engine. The reference implementation’s pinned MX Spike model matched every
padded BF16 output against the independent, pinned Radiance `mx_golden.cpp`
arithmetic reference.

| Model | Source PV B×M×N×K | Real head values | Padded BF16 values checked | Mean absolute error vs PyTorch f32 |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 32×8×64×8 | 512 | 1,024 | 0.0208 |
| DeepSeek-R1-Distill-Qwen-1.5B | 12×4×128×4 | 512 | 2,048 | 0.0171 |
| Gemma 2 2B | 8×4×256×4 | 1,024 | 4,096 | 0.0260 |
| SmolVLA | 12×16×64×16 | 1,024 | 1,024 | 0.00617 |

The complete K axis of each selected head is present after padding. The f32
error is a quantization diagnostic, not an accepted model-quality threshold.
The PV input already contains PyTorch's attention probability values; this
program does not compute QK, scale, mask, or softmax. The separate
[QK head replay](../model2mlir_four_model_fp8_qk_vpu_a042643_266c593/README.md)
executes each model's first QK head. Those programs are **not connected**;
full attention, other heads, checkpoint weights, and full-model inference
remain open. Gemma uses the installed default configuration because the gated
checkpoint configuration was unavailable offline.

Each model directory preserves its input provenance receipt, packed code and
scale bundle, independent golden, isolated model2MLIR capture, typed and
payload-bound MX MLIR, compressed physical program, data-free object receipt,
Spike receipt, and output log. The object and ELF SHA-256 values bind the
generated executables without committing toolchain-specific binaries.

Run the archive checks with:

```sh
python -m pytest -q tests/test_four_model_fp8_attention_pv_evidence.py
```

To regenerate operands, use `tools.prepare_tinyllama_attention_head_inputs`
for TinyLlama or `tools.prepare_model_attention_head_inputs` for the other
models, passing `--stage pv`, the corresponding archived full graph, and the
pinned model2MLIR source. Then run `tools.qualify_model_attention_head` with
`--stage pv`, the saved arrays, source region, batch index zero, K tile 32,
pinned MXQuant and Radiance arithmetic sources, profile, RTL and RISC-V
toolchains, and `--run-spike`. The source regions are `matmul_5` for the three
language models and `matmul_4` for SmolVLA. Float `.npy` arrays stay outside
the repository; packed bytes and source digests preserve the executed data.
