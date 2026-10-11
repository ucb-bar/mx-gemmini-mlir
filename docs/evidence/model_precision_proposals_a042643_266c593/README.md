# Four-model MX precision proposals

These eight JSON files come from `tools.plan_model_precision` applied to the
archived model2MLIR `a042643` portable graphs and pinned Gemmini RTL
`266c593`. Each file binds the exact frontend graph SHA-256 and target profile
SHA-256. The two profiles are `MxE4M3Fp4VpuGemminiRocketConfig` (MX+VPU) and
`MxGemminiRocketConfig` (MX without VPU).

The current requested policy is FP8 E4M3 with E8M0 scales as the general MX
baseline, FP4 E2M1 with E8M0 scales for aligned rank-two matrix
multiplications (including Q/K/V/O linear projections), and FP8 E4M3 for
exposed attention QK/PV contractions. The plan records a source operation's
region and module, making its precision decision auditable. Bias additions,
rotary transformations, and noncontraction operators still need a host,
Muon, or VPU lane. No FP6 selection is made.

| Model | FP4 rank-two sites | FP8 QK/PV sites | Other/host contraction regions |
| --- | ---: | ---: | ---: |
| TinyLlama, 2 random layers | 15 | 4 | 1 |
| DeepSeek-R1-Distill-Qwen-1.5B, 1 random layer | 8 | 2 | 4 |
| Gemma 2 2B, 2 random layers | 15 | 4 | 1 |
| SmolVLA, reduced random model | 81 | 24 | 93 |

These counts are for archived reduced graphs, not pretrained full-depth
models. A JSON `selected_format` is a proposed placement. It does not assert
that a site has quantized operands, a runnable kernel, or numerical quality.
The FP8 QK/PV sites need a batched lowerer; ordinary FP4 sites need complete
data binding, scheduling, and integration with their surrounding graph.
The FP4 projection outputs are BF16, so Q and K must be explicitly converted
to MXFP8 with E8M0 scales before QK. Softmax probabilities and V need the
same conversion before PV. The plan records these operand boundaries; no
implicit FP4-to-FP8 reinterpretation is allowed.

To regenerate one proposal with the pinned RTL checkout:

```sh
python -m tools.plan_model_precision \
  --model-mlir docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/pinned/gemmini-mx-cleanup \
  --out /path/to/new/tinyllama.precision.json
```
