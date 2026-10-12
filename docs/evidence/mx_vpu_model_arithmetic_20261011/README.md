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

The latest linked first-stage builds are summarized in `index.json`. Counts
are distinct source regions, not dynamic invocations. MX counts are selected
contractions; VPU counts are lowered source regions in the final Merlin build.
The MLIR captures retain host operations without a qualified MX/VPU mapping,
including indexing, casts, comparisons, selection, and trigonometry. The VPU
extension does not provide a direct sin/cos/select operation. The source-only
decode rewrite probes in `index.json` prove pattern coverage on those captured
graphs, but are not new linked or executed decode programs. SmolVLA action
decode has no MX/VPU candidate and uses Merlin's host route.

Nicolas's pinned MX Spike extension executed standalone generated VPU objects
for softmax, broadcast and scalar binary arithmetic, mixed BF16/f32
arithmetic, GELU, `10000^x`, and signed row reductions. `index.json` records
the precise result and SHA-256 of each passing receipt. These tests compare
BF16 outputs with independent scalar references at the tolerances stated in
the receipts. They do not qualify complete model outputs. In particular,
`10000^x` passes a 5% relative and 0.003 absolute criterion on the sampled
rotary exponent range, with observed maximum relative error 3.1963% and
maximum absolute error 0.002175. GELU observed maximum absolute error
0.011774 under its 2% absolute-scaled criterion. These tolerances are not
yet a model-level quantization acceptance policy.

The current status remains `linked_experimental_numeric_qualification_pending`:
complete quantized inference sessions have not been compared against an
independent PyTorch reference. Prior whole-session Spike evidence predates
this VPU rewrite. This archive is a coverage and isolated arithmetic milestone,
not a claim of full-model numerical correctness or performance parity with
Nicolas's handwritten kernels.
