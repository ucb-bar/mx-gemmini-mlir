# Source-bound MX+VPU model arithmetic (2026-10-11)

This archive records the MX+VPU routing of operations in the four full
checkpoint model2MLIR captures. The selected profile is
`MxE4M3Fp4VpuGemminiRocketConfig` at SHA-256
`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`.
The target package implements Merlin's `DeviceRouting` catalog builder and
post-offload transform. Merlin owns graph partitioning, host lowering, and
linking; the target package owns MX legality, code generation, and VPU source
patterns. `vpu_residual_eligible` reparses the final prepared MLIR and rejects
a stage if a selected VPU operation remains there.

The latest linked stage builds are summarized in `index.json`. Counts
are distinct source regions, not dynamic invocations. MX counts are selected
contractions; VPU counts are lowered source regions in the final Merlin build.
Every row below has exact MX catalog coverage, zero host softmax regions, and
zero final-MLIR operations matched by the five VPU selectors.

| Captured stage | MX contractions | Softmax | Add | Other binary | Unary | Row reduce |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama prefill | 200 | 22 | 88 | 358 | 156 | 45 |
| DeepSeek Qwen prefill | 254 | 28 | 196 | 454 | 198 | 57 |
| Gemma 2 prefill | 236 | 26 | 104 | 686 | 289 | 131 |
| SmolVLA prefix | 243 | 28 | 161 | 623 | 151 | 95 |
| SmolVLA flow | 148 | 16 | 60 | 352 | 107 | 33 |
| TinyLlama decode | 200 | 22 | 88 | 358 | 156 | 45 |
| DeepSeek Qwen decode | 254 | 28 | 196 | 454 | 198 | 57 |
| Gemma 2 decode | 236 | 26 | 104 | 686 | 289 | 131 |

The MLIR captures retain host operations without a qualified MX/VPU mapping,
including indexing, casts, comparisons, selection, and trigonometry. The VPU
extension does not provide a direct sin/cos/select operation. The source-only
decode rewrite probes in `index.json` prove pattern coverage on those captured
graphs. All three language-model decode stages also have newly linked ELFs.
SmolVLA action decode has no MX/VPU candidate and uses Merlin's host route;
its earlier linked Spike run matched all 1,600 elements of the first captured
result against unquantized PyTorch exactly. That is one isolated stage result,
not the complete SmolVLA session.

The reference implementation’s pinned MX Spike extension executed standalone generated VPU objects
for softmax, broadcast and scalar binary arithmetic, mixed BF16/f32
arithmetic, GELU, `10000^x`, and signed row reductions. `index.json` records
the precise result and SHA-256 of each passing receipt. These tests compare
BF16 outputs with independent scalar references at the tolerances stated in
the receipts. They do not qualify complete model outputs. In particular,
`10000^x` passes a 5% relative and 0.003 absolute criterion on the sampled
rotary exponent range, with observed maximum relative error 3.1963% and
maximum absolute error 0.002175. GELU observed maximum absolute error
0.011774 under its 2% absolute-scaled criterion. Gemma's source-bound logits
`tanh` now uses a VPU sigmoid sequence; its f32 and BF16 128-element Spike cases
observed maximum absolute error 0.010121 under the 0.02 bound. These tolerances are not
yet a model-level quantization acceptance policy.

The current status remains `linked_experimental_numeric_qualification_pending`:
complete quantized inference sessions have not been compared against an
independent PyTorch reference. Prior whole-session Spike evidence predates
this VPU rewrite. This archive is a coverage and isolated arithmetic milestone,
not a claim of full-model numerical correctness or performance parity with
the reference implementation’s handwritten kernels.
