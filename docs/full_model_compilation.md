# Full-model compilation with MX Gemmini + VPU

The [compilation guide](compile_e2e_models.md) gives commands to capture,
compile, link, and run the four checkpoint-backed sessions. This page records
the scope and evidence behind that workflow. The selected target is
`MxE4M3Fp4VpuGemminiRocketConfig` from the pinned Gemmini MX RTL profile.

## Compiler ownership

| Component | Responsibility |
| --- | --- |
| model2MLIR | Capture the source model as typed MLIR, external weights, inputs, reference outputs, and a multi-program session contract |
| Merlin | Parse and partition the complete graph; lower unmapped operations to compiled RV64 host code; generate argument, weight, state, and session ABIs |
| MX Gemmini target package | Select legal FP4/FP8 contractions and VPU source patterns; emit command objects and host ABI rewrites; verify exact source-region coverage |
| MX Gemmini dialect | Represent and verify typed target operations and explicit MX-to-Muon handoff contracts |

The full-model path uses Merlin's `DeviceRouting` catalog and post-offload
transform. It does not feed an entire captured model directly to
`mx-gemmini-opt`. The C++ dialect remains the typed interface for target
operations and source-bound kernel compilation; the catalog integrates
executable target objects into a whole-program build.

## Linked checkpoint sessions

The [four-session archive](evidence/four_vpu_session_links_20261012/README.md)
records standalone RV64 ELF links from complete pretrained checkpoints and
captured synthetic inputs. The stage and session linkers verified capture
receipts, exact selected MX regions, VPU rewrite coverage, object and profile
identities, shared checkpoint bytes, and state bindings.

| Model | Distinct program graphs | Selected MX contractions per graph | Session steps |
| --- | --- | --- | --- |
| TinyLlama | prefill, decode | 200, 200 | 1 + 32 |
| DeepSeek-R1-Distill-Qwen-1.5B | prefill, decode | 254, 254 | 1 + 32 |
| Gemma 2 2B | prefill, decode | 236, 236 | 1 + 32 |
| SmolVLA | prefix encode, flow denoise, action decode | 243, 148, 0 | 1 + 10 + 1 |

These are static source-region counts for each compiled graph; recurrent
steps invoke the same compiled program again. The selected profile uses FP4
for eligible ordinary matmuls and FP8 for eligible attention and rotary
contractions. The [stage archive](evidence/mx_vpu_model_arithmetic_20261011/README.md)
lists the source-bound VPU softmax, add, other binary, unary, and row-reduction
counts. Every device stage has zero operations left in its final MLIR that
match the current MX/VPU selectors. Other operations, including indexing,
casts, comparisons, selection, and trigonometry, remain in Merlin's compiled
host path. SmolVLA action decode has no selected MX/VPU operation.

The session linker uses Merlin's target-neutral session generator to bind
prefill/decode or prefix/flow/action state and materialize one verified
checkpoint-weight blob per session. Each recurrent invocation resets the
temporary guest arena while persistent state remains live. A successful link
has status `linked_experimental_numeric_qualification_pending`.

## Execution and numerical evidence

The [full-checkpoint TinyLlama decode stage](evidence/tiny_vpu_decode_spike_20261012/README.md)
completed one captured input step on the pinned MX+VPU Spike extension. It
reported guest `DONE`, zero memref rank mismatches, and validated digests for
logits, KV cache, and position state. It is an integrated stage execution,
not a complete recurrent session.

The [matched two-layer control](evidence/tiny_two_layer_vpu_isolation_20261012/README.md)
keeps the captured graph and all 20 MX contractions fixed while changing the
VPU rewrite. Both variants complete on Spike; their logits and KV-cache
digests differ. The [full-checkpoint operand diagnostic](evidence/tiny_decode_operand_fake_quant_20261012/README.md)
reproduces the saved unquantized PyTorch decode output and exercises the
selected FP4/FP8 operand policy, including the padded rotary K=1 BMM. That
PyTorch proxy uses different matrix and VPU arithmetic from the hardware.

The [SmolVLA full-session timing run](evidence/smolvla_vpu_full_session_timing_20261013/README.md)
completed all 12 steps and 45 expected events on Nicolas's MX+VPU Spike with
the `e4m3_vpu` preset and `mem.host_tracking=0`. It provides a stage-level
modeled cycle baseline, but its output digests are not an independent numerical
oracle. Complete Spike session receipts for the other models and an independent
full-graph reference under the exact FP4/FP8 conversion and MX/VPU BF16
arithmetic are still needed to qualify numerical correctness. Model quality
and performance relative to handwritten kernels require separate workload
inputs, output/state comparisons, schedules, and measurements. The selected
[Merlin target contract](../contracts/target_contract.yaml) therefore remains
`structural_unqualified` with `phase0_admitted: false`.

## Other configurations and kernel-level coverage

The [profile guide](current_mx_profiles.md) describes independent MX
configurations, precision modes, VPU capability, and mesh geometry. The
end-to-end commands above qualify the named MX+VPU profile only; selecting
another profile reruns legality, code generation, and numerical validation.
The [source-kernel coverage map](nicolas_application_kernel_coverage.md)
and [physical compiler record](compiled_mx_pipeline.md) contain the
source-bound FP4, FP6, FP8, VPU, and resident-kernel results. Kernel-level
comparisons do not substitute for a whole-model numerical check.
