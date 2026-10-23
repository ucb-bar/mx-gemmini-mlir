# Model artifact audit, 19 October 2026

The audit passed for the three published completed workloads: TinyLlama's
160-token forward, its prefill-plus-32-decode session, and SmolVLA's complete
prefix/flow/action session. It reopens execution artifacts and generated
compilation output. It does not certify arbitrary inputs or FPGA performance.

The retained [machine report](evidence/model_release_audit_20261019/audit.json)
and adjacent verifier logs record the checks. The
[release registry](../examples/models/release.json) supplies the model,
checkpoint, policy, tool, and executable identities. Run the audit again with:

```sh
PYTHONPATH=/path/to/merlin/src:/path/to/model2MLIR:. \
  python -m tools.audit_published_models --out /path/to/new-audit \
  --build-root /path/to/retained-builds \
  --radiance-root /path/to/radiance-kernels
```

The last two arguments are optional. Without the build root, the tool checks
the portable archives and their recorded executable bindings; it does not
claim to have reopened executable bytes that are not distributed. The
published audit did reopen those original large binaries and stage objects.
No simulation was rerun as part of that artifact check.

## What passed

| Check | Scope |
| --- | --- |
| Archive integrity | Hash-pinned dependency chains, manifests, member sizes and hashes, safe paths, frozen producer sources, generated IR/catalogs, commands and receipts |
| Actual binaries | Both sides of the model comparisons: complete ELF hashes, allocated section observations and retained stage objects |
| Output protocol | Terminal completion, complete typed output streams, output/state digests, and complete tensor bytes; no accepted timing-model deadlock |
| TinyLlama forward | All 5,120,000 F32 words exact against the corrected compiler baseline; unchanged source-relative 33-position quality gate |
| TinyLlama cached | All 99 output/state digests and 33 complete logit tensors exact; floating-reference diagnostic unchanged |
| SmolVLA | All 33 output/state digests and 11 complete action tensors exact; original floating-reference diagnostic preserved |
| Precision and placement | Published policies equal actual build probes; catalog bindings all FP8; host/VPU site counts recomputed from policy |
| Timing | Actual console intervals reconcile with receipts; reset and state-handoff accounting included for sessions; simulator/profile identities checked by model readers |
| Paired attention | Retained first-step and final-step source/compiler head outputs and paired timing archives independently checked |
| Radiance attribution | Comparison receipt pins the correct repository and excludes invalid cross-simulator cycle ratios; all seven listed GEMM/GEMV driver hashes reopened; retained command/issuer/driver bytes, traffic proofs and output-comparison observations checked |
| Reproduction helpers | Real exports of all three captures with matching external checkpoint files; complete output checker exercised against all three completed runs |
| FireSim packaging | Actual cached ELF hash/load segments/entry checked, derived 8 GiB memory requirement, packaged workload resolved by the pinned FireSim parser |

The FireSim check is a workload-format and executable-packaging check. It does
not establish that the host-target console/exit transport or the executable
has run on an FPGA. [Packaging requirements](compile_e2e_models.md#package-a-session-for-firesim)
are part of the deployment handoff.

## Corrections made while preparing the release

The old public guide described the default mixed FP4/FP8 policy. The actual
completed results use all-FP8 offloaded matrices; the release now exports
those exact per-stage policies. Sensitive softmax/elementwise placement is
explicit rather than inferred from the policy name.

The independent full TinyLlama program and timed SmolVLA heads come from
`npu-exploration`. The Radiance GEMM/GEMV/GQA controls come from
`radiance-kernels` at the pinned revision. They are documented separately.
Historical “stitched” model controls kept the compiler's host/VPU graph and
substituted native MX issuers; they were not independent Radiance model
applications.

SmolVLA's retained reference preserves its native BF16 backbone. Describing
it as an entirely FP32 model was inaccurate, even though the action arrays
are F32. The current documentation uses “floating-point reference.”

The cached compact-RHS execution has completed and is now included. Its
actual guarded decoder and all complete outputs were checked. The older
compact forward executable predates that privacy guard; a later source
reanalysis does not change the historical binary.

The local Radiance checkout lacked the historical Git object, but all seven
measured driver files matched the pinned receipt byte-for-byte. The audit
reports driver-byte correspondence, not whole-checkout identity. The pinned
Gemmini configuration source was also retrieved and matched the target
profile's recorded source digest; an unrelated older local checkout was not
used to infer hardware compatibility.

## Complete Radiance-derived forward audit

The [Radiance Rocket port evidence](evidence/radiance_rocket_forward_20261019/README.md)
adds an independent full-forward comparison on the same Rocket MX+VPU target.
Its reader checks all 160 source and compiler logit rows, both the late-33
and complete-160 numerical gates, descriptor-derived accelerator work, and
unchanged outputs under diagnostic profiling. This is separate from the
older `npu-exploration` control above.

The [optimization archive](evidence/radiance_rocket_optimization_20261019/README.md)
retains the subsequent complete forward experiments, including numerical
failures and a softmax fusion regression. Its reader recomputes both numerical
gates from all observed logits, checks the timing/work counters, and compares
fusion receipts with actual calls in the lowered LLVM. An artifact
verification pass means these observations and bindings agree; it does not
mean that a candidate passed the numerical gate.

The FP32 host-softmax/residual execution with a finite-input maximum and RNE
underflow shortcut passes all six numerical checks at **6,264,147,801 cycles**,
versus **7,082,331,012** for the port: **1.131× speedup**, or **11.55% fewer
cycles**. All 5,120,000 output words match both its 6,997,758,443-cycle
maximum-only control and the preceding 7,087,996,683-cycle passing control.
Matrix/VPU work and transfer bytes are identical across those three runs.
The underflow receipt and actual LLVM bind all 22 conditional exponential
paths, their -104 cutoff, positive-zero result and RNE/unobserved-status
contract. The separate Torch CPU arithmetic run
takes 6,435,411,995 cycles and fails both top-token agreement limits. The
FP32-host-softmax/BF16-residual alternative fails two numerical limits.

The separate polynomial-softmax integration exposed a dense-float
serialization error in entry projection. Merlin revision
`f8a1464c9` fixes that writer by using the shared portable representation.
Regression tests preserve F32 and F64 infinities, NaN payloads and signed
zeros through projection and reparse. Initial incomplete executions stopped
at the first probability packer and supply no full-forward timing result.

The learned-weight audit reopens the actual source and compiler executables,
undoes the source's combined/tiled layouts, and checks all 155 FP8 code and
E8M0 scale payloads in canonical coordinates. Those encoded matrices match.
Activation precision, reduction order, rotary coefficient arithmetic and
matrix schedules remain explicit implementation differences.

The [FP32 RMS packing probe](evidence/f32_rms_packing_20261019/README.md)
has a separate complete-byte audit. It checks actual ELF constants and all
target packed outputs against FP32 normalization and a retained independent
scalar quantizer. Its local cycle savings are not substituted for full-model
execution measurements.

The [linked payload checker](../tools/audit_merlin_elf_payloads.py) can also
reopen a compiler ELF and check its actual token IDs, captured constants,
all fixed matrix codes/scales, and retained embedding/normalization weights
against the supplied capture and compiled catalog. It supplements executable
hashes with checks of the bytes the executable will read; it does not run the
model or establish timing or numerical quality.

```sh
python -m tools.audit_merlin_elf_payloads \
  --capture "$ARTIFACT_ROOT/tinyllama-forward-inputs/capture" \
  --build "$ARTIFACT_ROOT/tinyllama-forward-exp-build" \
  --out "$ARTIFACT_ROOT/tinyllama-forward-linked-payloads.json"
```

## Limits of the result

The numbers are modeled Rocket-Spike cycles with `mem.host_tracking=0`.
No matching FPGA run or bitstream is qualified. The older TinyLlama source
distributions cover 33 positions; the newer Radiance-derived Rocket port and
compiler both retain all 160. The
[complete independent cached comparison](evidence/radiance_rocket_cached_20261023/README.md)
finds the compiler slower and failing the worst-row KL limit. Complete SmolVLA
still has no independent full source-session performance baseline. SmolVLA has no robot-task acceptance threshold. Synthetic inputs
do not establish dataset accuracy or perplexity.

External checkpoint and toolchain files are still required for compilation.
The release records their identities and checks supplied files, but the
original Python wheel provenance and complete hermetic build environment
were not retained. Build-command plans and input exports have been checked;
the new public helper has not rerun all full model compilations/executions.
Historical source/object/ELF bindings establish the published measurements;
fresh outputs must pass the complete-output checker before being called a
reproduction. Stable legacy filenames inside evidence remain unchanged to
preserve links and hashes.
