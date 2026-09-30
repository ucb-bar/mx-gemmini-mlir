# MX Gemmini Phase 0 review

Review snapshot: 2026-09-30. The selected software contract remains
`unreviewed`. This review applies to the exact Gemmini
`f0167390b56fb315deea90ac1fc3983772e92d82`, MxGen
`a27ce3cd81513210c21f971ec3977defd13fa21e`, and
`GemminiMxFPConfigs.standaloneMxFPConfig`. The RTL at those revisions is the
authority for the selected configuration.

## Results

| Area | Result | Boundary |
| --- | --- | --- |
| Source selection | `rtl_check` matches the selected revisions, the seven pinned source files, format codes and widths, max finite values, BF16 readout code, block-scale floor 104, and the selected rounding paths. | Source cross-check; not a complete elaboration closure. |
| Operand conversion | Differential comparison against [`microscaling-quant` at `ea3f5bb`](https://github.com/chloe-wong/microscaling-quant/tree/ea3f5bb) used RNE, the 2^-23 hardware scale floor, and E3M1 before FP4. For each of MXFP8, MXFP6, and MXFP4, the 65,280 finite BF16 bit patterns had identical E8M0 scales and numerical BF16 dequantized values. Random, zero, and edge-value blocks also matched. | One FP8 and one FP6 negative-zero sign differed from the independent model; FP4 had 288 zero-sign differences and no nonzero differences. The RTL's `E3M1Tofp4` canonicalizes zero to positive zero. These comparisons do not check mesh arithmetic or all block combinations. |
| Capture handoff | Mixed FP8/FP6/FP4 sites and one resident output chain pass model2MLIR capture and the MX dialect verifier. | This checks identity and operation structure, not executable lowering or whole-model accuracy. |
| Numerical review | The fake-quant kernel models BF16 operand conversion. | Mesh product truncation, the 16-lane reduction schedule, FP6 LUT choice, nonfinite poisoning, output requantization, host operations, and full-model accuracy still need selected-oracle evidence. |

The available three-stage SmolVLA full-checkpoint diagnostics use synthetic
inputs and their stage receipts report `source_closure_verified: false`.
Their materialized files are useful for structural inspection, but they do
not supply the selected, attributed MX application capture needed below.
One eager denoise step from the pinned pretrained checkpoint with synthetic
inputs observed 302 of 303 Linear modules. Its dimension-only screen counted
73 sites within the FP8, FP6, and FP4 tile bounds; 229 failed activation
M/rank or N/K bounds. This is a shape diagnostic,
not a quantized model capture, accuracy result, or admitted application corpus.
An earlier FP8-only OOT adapter also completed model2MLIR FX capture of that
synthetic denoise step on the older active contract: its manifest reports 73
quantized Linear sites, 24 quantized QK/PV sites exposed from 12 masked SDPA
calls, 230 skipped Linear modules (including one absent from the exported graph), and 64
skipped functional sites. The importer reported 9,735 decomposed operations and
zero opaque operations. This is capture and site coverage only; it has no
source-closed input corpus, full-model accuracy result, or accelerator execution.
The full capture's contract and policy digests, 391-site manifest, and source
MLIR passed the OOT handoff validator; `mx-gemmini-opt` accepted the rendered
operation plan. This verifies the dialect boundary for the diagnostic capture,
not executable MX lowering. These counts predate contiguous operand enforcement
and require a fresh capture before they describe the current adapter.
The graph-first adapter now preserves source lineage through TorchAO Linear
selection, functional Q/DQ, and re-export. The same synthetic denoise-step
capture has a complete original-to-quantized-to-prepared frontend trace and
no trace blockers; its manifest is digest-bound to the MLIR module. This
closes the structural provenance gap for this capture. The inputs remain
synthetic, the contract remains on the older RTL pin, and the trace does not
establish numerical equivalence or accelerator execution.

The current [`gemmini-mx-cleanup` head at `2029218`](https://github.com/ucb-bar/gemmini/commit/2029218197f771ce71416f859d975bea47b7aabc)
is seven commits after the selected pin and selects MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`. It adds a separate E4M3
single-format configuration. It also changes the shared controller and loop
paths: scale-load funct 27 now has pitch, destination, row-count, and gating
fields; loop-managed scale commands 31 and 32 were added; requantizer LUT
construction is conditional. The selected standalone configuration still has a
LUT. The [isolated latest-RTL diagnostic](latest_rtl_diagnostic.md) builds a
matching simulator and checks explicit and loop-managed scale commands. The
TorchAO operand kernel now accepts this exact Gemmini/MxGen pair for explicit
candidate capture, while rejecting other revision pairs and configurations.

## Current RTL source audit

The candidate source record is
[`rtl_candidate_2029218.yaml`](../mx_gemmini_support/contracts/rtl_candidate_2029218.yaml).
The corresponding
[`software-spec-2029218-candidate.yaml`](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml)
pins those revisions and declares the new scale-load and loop-management
fields. It passes Merlin software-spec validation and the package's source
hash, format, and rounding cross-check. The source check now also compares
the funct IDs, funct-27 command slices, `CONFIG_SCALE_MEM` bit references,
and zero/contiguous-row expressions with the candidate declaration. The source
check is separate from the bounded executed command checks linked above.
Its status remains `unreviewed`; the Merlin support provider and default
contract still identify the older revision. The candidate must be selected
explicitly for TorchAO capture. The older simulator digest does not apply to
this revision.
The candidate operand kernel also passed the pinned `microscaling-quant`
[comparison](evidence/operand_review_2029218.json) for all finite BF16
patterns and the documented random, zero, and edge blocks: zero numerical or
E8M0 scale mismatches for each MX format.
Zero-sign differences remain as described above. Mixed-format handoff and the
four synthetic candidate workload [captures](evidence/iteration_capture_2029218.json)
pass with the candidate contract;
these are structural capture results, not model accuracy or accelerator runs.
The source record's 12 file hashes match the checked-out `2029218` Gemmini tree and its
`56ef1c6` MxGen submodule. This is a scoped source census, not the complete
elaboration closure. The selected
`GemminiMxFPConfigs.standaloneMxFPConfig` still has a 16 by 16 weight-stationary
mesh, 32-element scale blocks, a LUT, 256 KiB scratchpad, 64 KiB accumulator,
and two accumulator banks. It sets `has_nonlinear_activations = false` and
`has_normalizations = false`; the current software contract assigns those
operations to the host. These settings were also used in the isolated
elaboration; their numerical implications remain unreviewed.

The changed protocol must be part of any candidate software contract for this
revision:

| Instruction | RTL fields and behavior |
| --- | --- |
| `MX_LOAD_SCALES` (funct 27) | `rs1[39:0]` physical source address; `rs1[63:40]` source row pitch; `rs2[31:0]` bytes per row; bit 32 selects A/activation (0) or B/weight (1); bits 45:33 select destination scale-memory byte offset; bits 53:46 give row count; bit 54 gates the load until its destination half is free. Zero row count means one row and zero pitch means contiguous rows. Address, pitch, and destination must be eight-byte aligned. The loader truncates the byte count down to an eight-byte multiple. |
| `LOOP_WS_CONFIG_SCALES` (funct 31) | `rs1` and `rs2` provide A and B scale source addresses. A zero A address selects the legacy path without loop-managed scale loads. |
| `LOOP_WS_CONFIG_SCALE_STRIDES` (funct 32) | `rs1` and `rs2` provide A and B scale row pitches. The loop emits gated two-dimensional funct-27 loads into the half indexed by its loop slot. |
| `CONFIG_SCALE_MEM` (funct 26) | `rs2[16]` waits for selected loads to land; `rs2[17]` waits for loop-managed halves to be ready; `rs2[18]` permits resident A-scale reuse. The execute controller checks these conditions before accepting the config. |

A software encoder for this revision must reject nonzero byte counts that are
not multiples of eight, rather than silently accepting the RTL's truncation.
The candidate-only `scale_load_2d_operands` encoder checks the exact source
pins and declared fields, and rejects unaligned source, pitch, and destination
values, source address overflow, more than 255 rows, and writes crossing a
4 KiB destination half. Its emitted fields were used in four passing explicit
load diagnostics. Loop-managed FP8 passed two tests, but loop-managed FP6 and
FP4 each produced 768 BF16 mismatches in the 32-square test. The candidate
therefore restricts loop-managed scales to FP8 and requires explicit funct-27
loads for FP6/FP4.
The same latest simulator also passed one 64×64×64 explicit-load program per
format with zero BF16 mismatches. This strengthens the bounded fallback path;
the general scheduler and larger shapes remain unqualified.

The source audit also finds a shared `GemminiConfigs.scala` change to DMA
column-field sizing and optional LUT wiring in `MxRequantizer.scala`. The
optional-LUT change does not remove the LUT from the selected standalone
configuration. The new build and scale-load checks do not exercise these
changed shared paths sufficiently to move the active pin.
The current `scale_load_rs2` packer encodes the older one-dimensional instruction
and stays tied to the older contract.
The selected MxGen submodule's `MxFpMul_MxGemmini_BF16Out_Spec` passed its one
self-checking PE test suite at this revision. That is unit evidence for the PE
cases in the suite, not a Gemmini integration or simulator verdict. The
standalone Gemmini Scala test could not compile with the checkout's existing
build dependencies, so it supplies no additional result.

[`pi0-quant`](https://github.com/chloe-wong/pi0-quant) and
[`smolVLA-quant`](https://github.com/chloe-wong/smolVLA-quant) currently describe
per-tensor power-of-two FP8 model experiments. They are useful for workload
coverage and accuracy questions, but do not establish this contract's per-32
MX scale rule or its FP6/FP4 behavior. `microscaling-quant` supplies the
operand comparison above; its documented hardware-matched systolic result is
for MXFP8 E4M3, not a qualification of every format and model here.

Reproduce the operand comparison with a clean checkout of the stated
`microscaling-quant` revision in the same Python environment as this package:

```sh
python -m mx_gemmini_support.rtl_check /path/to/pinned-gemmini
python -m mx_gemmini_support.review_operands /path/to/microscaling-quant
```

The source check rejects a changed Gemmini or MxGen revision or any changed
pinned source file. The operand comparison rejects a changed reference revision
or locally changed `mxq` source and prints per-format, per-case counts.

## Phase 0 admission work

1. Freeze the intended RTL revision. For `2029218`, enforce explicit FP6/FP4
   scale loads or resolve their failed loop-managed path in RTL, check repeated
   half reuse and changed shared paths, review source/toolchain binding, and
   rerun the full format and chain diagnostics.
   Keep the existing `f016739` receipts attributed to their original revision.
2. Select exact MX application captures and a versioned site policy. Review any
   FP6 codebooks, host placement, and accuracy choices with the numerical owners.
   The example FP8-only policy is a bringup policy, not a reviewed whole-model
   selection. The [candidate iteration roster](iteration_roster_candidate.md)
   proposes four independent MX-sized captures and gives their artifact
   materialization command; it has not been admitted.
3. Derive a new MX-only conformance requirement from those captures. The retained
   synthesis sidecar is `unverified_legacy` and contains BF16/int8 accelerator
   cells outside this three-format contract. Supply a same-target capability
   contract and selected RTL facts, then regenerate and review a digest-bound
   synthesis profile under the run artifact root.
4. Run Merlin Phase 0 preflight on those selected inputs, then generate capsules
   and check independent L0/L1 numerics plus source-bound Spike L2 and Verilator
   L3. Review host transfers and the complete census before admitting a corpus.

The current `mx-gemmini-functional --phase 0` preflight reports
`configuration_ready: false` because the selected synthesis profile is
`unverified_legacy`. No Phase 0 admission or functional compiler certificate is
claimed by this review.
