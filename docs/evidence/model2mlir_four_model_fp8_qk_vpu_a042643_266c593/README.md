# Four model-derived FP8 attention QK heads on pinned MX Spike

This archive executes **one head from the first QK batched matrix multiplication**
in each reduced model2MLIR graph. Operands came from a seeded forward pass of
each randomly initialized model and were tapped from its decomposed PyTorch
export graph. The exact source `linalg.generic` region and full model2MLIR
graph SHA-256 are recorded in each payload. SmolVLA's selected site is in its
first vision encoder attention layer.

Each head uses MXFP8 E4M3 for both operands, E8M0 scales per 32 K elements,
BF16 operand conversion, and the pinned `MxE4M3Fp4VpuGemminiRocketConfig`.
The generated RV64 object issues MX commands; the VPU is present in this
configuration but is unused by these heads. Nicolas's pinned MX Spike model
matched all 512 BF16 values in each padded output against the independent,
pinned Radiance `mx_golden.cpp` arithmetic reference.

| Model | Source QK B×M×N×K | Real head values | Padded BF16 values checked | Mean absolute error vs PyTorch f32 |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 32×8×8×64 | 64 | 512 | 0.345 |
| DeepSeek-R1-Distill-Qwen-1.5B | 12×4×4×128 | 16 | 512 | 0.237 |
| Gemma 2 2B | 8×4×4×256 | 16 | 512 | 0.645 |
| SmolVLA | 12×16×16×64 | 256 | 512 | 0.102 |

The K dimension is complete for the selected head. M is padded to 16 rows
and N to 32 columns for the physical program. The comparison covers the
entire padded output, including the real values above. The f32 error is a
quantization diagnostic, not an accepted model-quality threshold. Attention
scaling, masks, softmax, PV, other heads, full model routing, and checkpoint
weights remain outside these programs. Gemma uses the installed default
configuration because the gated checkpoint configuration was unavailable
offline. No complete model or end-to-end inference is claimed.

Each model directory preserves the input provenance receipt, packed code and
scale bundle, independent golden, isolated model2MLIR capture, typed and
payload-bound MX MLIR, compressed physical program, data-free object receipt,
Spike receipt, and output log. The object and ELF SHA-256 values bind the
generated executables without committing toolchain-specific binaries.

Run the archive checks with:

```sh
python -m pytest -q tests/test_four_model_fp8_attention_qk_evidence.py
```

To regenerate operands, use `tools.prepare_tinyllama_attention_head_inputs` for
TinyLlama or `tools.prepare_model_attention_head_inputs` for the other three,
with the corresponding archived full graph and pinned model2MLIR source.
Then run `tools.qualify_model_attention_head` with `--stage qk`, the saved arrays, source
region, batch index zero, pinned MXQuant and Radiance arithmetic sources,
profile, RTL and RISC-V toolchains, and `--run-spike`. The model-specific
source regions and K tile sizes are `matmul_4`/64 for TinyLlama,
`matmul_4`/128 for Qwen, `matmul_4`/256 for Gemma, and `matmul_3`/64 for
SmolVLA. Float `.npy` arrays stay outside the repository; the archived packed
bytes and source digests preserve the exact executed data.
