# TinyLlama forward placement policies

These configurations use the archived 160-token TinyLlama forward from the
[model build guide](../../../docs/compile_e2e_models.md). All offloaded
matrices use FP8 E4M3 with E8M0 block32 scales and BF16 matrix outputs.
The rotary-frequency singleton contraction remains FP32 on the host.
The files vary operator placement; they do not change matrix precision.

| Policy | Placement |
| --- | --- |
| `tinyllama-forward-hybrid-softmax.json` | Existing matrix/VPU placement, with all 22 softmax sites on the VPU |
| `tinyllama-forward-host-score.json` | Hybrid softmax, with the 22 scalar attention-score multiplications on the host |
| `tinyllama-forward-exp-vpu.json` | Softmax and sigmoid on the VPU; normalization, reductions, RoPE arithmetic, residual additions and other elementwise arithmetic on the FP32 host |
| `tinyllama-forward-exp-vpu-bf16-residual.json` | EXP-only placement, with all 44 residual additions on the BF16 VPU |
| `tinyllama-forward-host-softmax-bf16-residual.json` | All 22 softmax sites retain FP32 host arithmetic; sigmoid EXP and the 44 BF16 residual adds use the VPU |
| `tinyllama-forward-host-softmax-f32-residual.json` | FP32 host softmax and residual adds; sigmoid EXP uses the VPU |

For policies that retain VPU softmax, select `--vpu-softmax-implementation f32_reduce`.
This batches BF16 VPU exponentials and retains FP32 max, subtraction, ordered
sum and division. The default sigmoid retains BF16 arithmetic at each step.
`--vpu-sigmoid-implementation f32_reciprocal` is a separate numerical choice
that keeps FP32 arithmetic around the BF16 exponential.

The first two policies can use the original connected BF16 multiply and
transport options. The third uses host multiplies and normalization, so
omit the connected multiply, connected sigmoid, matrix/VPU transport and
RMS-packing flags. Projection grouping and private softmax/matrix streaming
can be measured separately under their documented legality conditions.
See [target and precision configuration](../../../docs/configure_target_and_precision.md)
for the flags and their arithmetic contracts.

Each JSON binds this captured MLIR:

`5c2cc98b8d0480893ebe6e682d094869e75a9dfc34c984e9930a269111be957a`

and this target profile:

`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`

Use the exact archived capture export for reproduction. For a new capture,
generate a new policy and identify its source operations again. The build
rejects stale or incomplete policies. Numerical and complete-model timing
results belong to the [comparison record](../../../docs/fair_full_model_comparisons.md);
selecting a policy alone does not establish a qualified speedup.

To compile the EXP-only configuration after exporting `tinyllama-forward`
with the model guide, use a fresh output directory:

```sh
"$CAPTURE_PY" -m tools.build_merlin_stage \
  --stage "$ARTIFACT_ROOT/tinyllama-forward-inputs/capture" \
  --workdir "$ARTIFACT_ROOT/tinyllama-forward-exp-build" \
  --profile "$MX_ROOT/profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json" \
  --merlin-src "$MERLIN_ROOT/src" --llvm-install "$LLVM_INSTALL" \
  --riscv-root "$RISCV_ROOT" \
  --execution-policy "$MX_ROOT/examples/models/optimization_policies/tinyllama-forward-exp-vpu.json" \
  --prepack-checkpoint-weights --prune-unused-checkpoint-weights \
  --share-activation-packing --integer-quant-groups \
  --host-math-guard "$ARTIFACT_ROOT/tinyllama-forward-inputs/guard.json" \
  --native-tile-policy movement --reuse-batch-rhs --compact-batch-rhs \
  --vpu-softmax-implementation f32_reduce \
  --structural-elf-admission --spike-arena-base 0xd0000000 \
  --compile-timeout-s 6000 --output-dump-cap 5120000 \
  --spike-timeout 43200 --spike-extlib "$SPIKE_EXTLIB"
```

This command selects the EXP-only placement with BF16 sigmoid arithmetic.
Use MX commit `157c8b07` or later, Merlin
`bb7c352a9e9c8ad085d24091fc83e4e7f5ff92ca`, and model2MLIR
`f5e5c674a585bebbc2507dbd028ebf9d26e2afbc`. Keep the model guide's exact
checkpoint, capture, target, toolchain and timing environment.

The separate transformations `--group-projections` and
`--fuse-softmax-packing` preserve the selected arithmetic. To evaluate the
FP32 sigmoid choice, add `--vpu-sigmoid-implementation f32_reciprocal`.
That choice also enables `--fuse-gated-projections` from MX commit
`129cd185cb4b3967feb4e3b91b9445fe1779dfe0`. Use separate fresh build directories
for each configuration and compare complete outputs. None of these options
changes the frozen release registry or its measured executable identities.

The BF16-residual variant uses the fourth policy in place of the EXP-only
policy above. Its 44 additions are selected uniformly: nonbroadcast adds
whose full result shape matches an RMS reduction input, here `1×160×2048`.
The source port also rounds its residual state to BF16. This is an explicit
numerical choice; no layers or tokens are selected using golden outputs.
The remaining host arithmetic and every matrix format are unchanged.

Add `--matrix-vpu-transport` to feed private BF16 matrix outputs directly
into those VPU residual adds. This direction is matrix-to-VPU. The separate
`--vpu-matrix-transport` option serves VPU-produced operands and is not used
by this host-normalization configuration. To combine BF16 residuals with
FP32 sigmoid and gated projection packing, use MX revision
`a4c2cfbb24b93a1a1978952a3d3d55baeb2e09f2` or later and add:

```sh
--matrix-vpu-transport --group-projections --fuse-softmax-packing \
--vpu-sigmoid-implementation f32_reciprocal --fuse-gated-projections
```

Check the actual fusion and transport receipts as well as complete output
quality and cycles. The first implementation of this combination retained
the unfused path because transport preceded gate/up matching; that no-op
control is kept separate from the corrected pass order.

Use MX revision `8df46f8a` or later for the softmax launch-count guard.
For the measured `32×160×160` attention shape, streamed packing would increase
EXP launches from 101 to 128 per layer. The guard retains the original hybrid
softmax path even when `--fuse-softmax-packing` is requested. The complete
unguarded run regressed by 16,207,172 cycles; the standalone probe did not
predict that result.

MX revision `272afe6d` adds `--fuse-f32-rms-packing` for closed host FP32 RMS
graphs. Sequential summation is the default. The explicit alternative
`--f32-rms-reduction striped16` changes FP32 reduction order to match the
Rocket port's lane summation. See the [arithmetic and legality contract](../../../docs/configure_target_and_precision.md#fp32-normalization-and-fp8-preparation).
Both choices need a fresh complete-model execution and unchanged numerical
gates. A local normalization probe is not a full-model speedup.

The following complete build command combines BF16 residual state, FP32 RMS
and the port's lane-ordered FP32 normalization choices. It requires MX
revision `cab1021e`, with the Merlin and model2MLIR revisions above. The
choice of gated division changes rounding relative to reciprocal/multiply.
Its timing and quality must come from its own complete execution.

```sh
"$CAPTURE_PY" -m tools.build_merlin_stage \
  --stage "$ARTIFACT_ROOT/tinyllama-forward-inputs/capture" \
  --workdir "$ARTIFACT_ROOT/tinyllama-forward-lane-order-build" \
  --profile "$MX_ROOT/profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json" \
  --merlin-src "$MERLIN_ROOT/src" --llvm-install "$LLVM_INSTALL" \
  --riscv-root "$RISCV_ROOT" \
  --execution-policy "$MX_ROOT/examples/models/optimization_policies/tinyllama-forward-exp-vpu-bf16-residual.json" \
  --prepack-checkpoint-weights --prune-unused-checkpoint-weights \
  --share-activation-packing --integer-quant-groups \
  --host-math-guard "$ARTIFACT_ROOT/tinyllama-forward-inputs/guard.json" \
  --native-tile-policy movement --reuse-batch-rhs --compact-batch-rhs \
  --matrix-vpu-transport --group-projections --fuse-softmax-packing \
  --fuse-gated-projections --vpu-sigmoid-implementation f32_reciprocal \
  --gated-activation-arithmetic divide \
  --fuse-f32-rms-packing --f32-rms-reduction striped16 \
  --vpu-softmax-implementation f32_striped16 \
  --structural-elf-admission --spike-arena-base 0xd0000000 \
  --compile-timeout-s 6000 --output-dump-cap 5120000 \
  --spike-timeout 43200 --spike-extlib "$SPIKE_EXTLIB"
```

The measured shape realizes 45 FP32 RMS fusions, 22 gated projection fusions,
22 grouped QKV completions and 44 matrix-to-BF16-residual handoffs. The
softmax launch-count guard retains the existing 22 batched hybrid softmax
calls. Check those counts in the rewrite receipts and the lowered LLVM
before interpreting the resulting execution as this configuration.

The two host-softmax alternatives, added in MX revision `5d32a8c1`, use the same matrix formats and select all
softmax sites uniformly. They retain the source graph's FP32 max, exponent,
sum and division on Rocket. To evaluate either one, replace the policy in the
command above and select `--vpu-softmax-implementation f32_reduce`. No VPU
softmax endpoint is then selected. For the F32-residual alternative also omit
`--matrix-vpu-transport`, since those residual consumers run on the host.
Keep FP32 RMS fusion and gated projection packing. These alternatives change
precision and placement and need independent complete-output quality checks.

MX revision `b4ade581` adds the independent schedule alternative
`--native-tile-policy movement_k256`. It caps FP8 K steps at 256 and permits a
complete aligned reduction such as K=160. It keeps the target profile and
matrix formats unchanged, while allowing a different accumulation order.
Use a fresh build and complete-output check for this option; the original
`movement` results do not establish its numerical behavior or performance.

To evaluate the explicit PyTorch 2.10 CPU arithmetic on either host-softmax
policy, add `--host-softmax-policy torch_2_10_cpu_softmax_f32_avx2`. This
selects all host softmax sites uniformly and requires finite input scores.
It uses Merlin's FMA polynomial exponential and eight-lane reduction order;
it changes numerical arithmetic and requires a fresh complete-output check.
The original host-softmax runs above retain scalar libm arithmetic. See the
[arithmetic contract](../../../docs/configure_target_and_precision.md#select-softmax-arithmetic)
before selecting this option.

To retain the original libm exponential, sequential sum and division while
specializing the maximum, select `--host-softmax-policy libm_f32_ordered_finite`.
Both host options require Merlin `94f747964881fddecb2808992c06f13cad451c2b`.
This includes the entry-IR serialization fix needed to preserve dense
floating constants through weight projection.
