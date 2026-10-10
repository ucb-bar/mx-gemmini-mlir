# MX Gemmini software contract and MLIR handoff

Licensed under Apache-2.0; see [LICENSE](LICENSE).

The MLIR dialect includes `readout_to_smem` and `wait` operations for
an explicit MX-to-Muon shared-memory handoff. The `radiance-mlir` composition
package verifies their ordering with Muon fences and barriers against a
selected SoC profile. General matrix and handoff lowering remains in progress.
The new physical `vpu_execute` and `spad_requant` operations lower to Rocket
RoCC commands for source-bound DIM16 MX+VPU profiles; see
[current MX profiles and VPU lowering](docs/current_mx_profiles.md).
`mx_gemmini_support.command_ir` represents checked physical command fields
and emits the same stream through Rocket RoCC or the Muon-side Radiance MMIO
gateway. The source specialization path binds packed operand bytes, E8M0
scales, and all FP6 A/B/C LUT lines to a profile-bound model2MLIR contraction.
New bound MLIR embeds a canonical resource manifest with each array's shape,
layout, byte count, and SHA-256. Typed `mx_gemmini.resource` results feed the
contraction's packed code and scale operands; `mx_gemmini.upload_lut` names
each LUT bank before compute. The dialect and physical lowerer check the
binding, and the external binary bundle supplies the bytes.
`physical_program.lower_bound_source` schedules configuration, scale/LUT DMA,
operand movement, K-wave compute, and BF16 or supported quantized readout for
complete output tiles.
`tools.qualify_source_mx` builds a standalone RV64 ELF and compares every
output on Nicolas's pinned Spike extension. The latest
[complete MX GEMM source roster](docs/compiled_mx_pipeline.md#latest-complete-radiance-mx-gemm-roster)
qualifies all 31 Radiance FP4, FP6, and FP8 drivers from model2MLIR captures:
23 drivers with BF16 output and 8 with requantized output match their source
goldens. The [one-command reproduction](docs/compiled_mx_pipeline.md#latest-complete-radiance-mx-gemm-roster)
rebuilds and checks those captures and Spike results against archived hashes.
The [plain MX Rocket profile test](docs/compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6)
also runs representative FP8, FP4, and FP6 source kernels on
`MxGemminiRocketConfig` without VPU. Two runs and a fresh-source rerun match
all 36,864 BF16 outputs and reproduce the generated ELFs on Nicolas's Spike.
The [selected-profile matrix](docs/compiled_mx_pipeline.md#selected-legal-modes-in-six-additional-mx-rocket-profiles)
qualifies 12 more FP8/FP4/FP6 cases across six named Rocket profiles, with
147,456 matching BF16 outputs and an explicit unsupported-mode rejection.
The [profile qualification catalog](docs/evidence/mx_profile_qualification_catalog_266c593/index.json)
now indexes direct stock-Spike execution receipts for all 40 named Chipyard
Rocket MX wrappers. An [eight-wrapper sweep](docs/compiled_mx_pipeline.md#direct-spike-receipts-for-the-remaining-rocket-wrappers)
closes the previous named-profile gaps with two source-derived runs per
wrapper, while the separate mode-class matrix records the stock Spike
weight-LUT discrepancy. The 41 Gemmini-only fragments remain structural
profiles.
A [fresh-checkout reproduction](docs/evidence/mx_fresh_checkout_205e178/index.json)
builds `mx-gemmini-opt` from the pushed branch and rechecks all 31 Radiance
MX GEMM drivers and eight Rocket-wrapper cases on Nicolas's Spike.
The same command now accepts the current Radiance `spatter-workloads` revision
`80f84ca` with `--compatible-source-revision`: all 31 driver and header bytes,
typed MLIR, physical streams, ELFs, and Spike logs match the pinned roster.
Its [compact receipt](docs/evidence/radiance_mx_gemm_80f84ca_upstream_repro_20261010/reproduction.json)
records the current source revision and exact baseline artifact hashes.
A [fresh published compiler run](docs/evidence/radiance_mx_gemm_fresh_4fc4d3a_82be2c7/index.json)
also reproduces all 31 drivers from upstream Radiance `82be2c7` on Nicolas's
Spike, comparing 466,944 output elements across BF16 and requantized cases.
The [source build-selection audit](docs/compiled_mx_pipeline.md#latest-complete-radiance-mx-gemm-roster)
finds that 18 of those named drivers are listed in Radiance's Makefile and 13
are source recipes excluded from its build.
The [read-once weight-stationary cases](docs/compiled_mx_pipeline.md#read-once-weight-stationary-radiance-mx-kernels)
also match the FP8 and FP4 down-projection source goldens and load each packed
weight tile once per K wave.
The [re-stream baseline](docs/compiled_mx_pipeline.md#four-pass-re-stream-comparison)
matches its committed source golden and shows the expected fourfold weight
traffic across four M tiles.
The [four-tile MX+VPU case](docs/compiled_mx_pipeline.md#four-tile-fp8-gemm-with-tilewise-vpu-epilogue)
captures `matmul * 2.0` through model2MLIR and connects contraction, VPU, and
readout in typed SSA. It applies one compiler-issued in-place VPU command
before each tile readout. All 65,536 BF16 outputs match
the source-derived ×2 reference on pinned Spike; this selected epilogue is not
yet a general graph lowering. The selected source `tk256` driver is excluded
from Radiance's 128 KiB build; this test uses Nicolas's 256 KiB MX+VPU profile.
A [current-source rerun](docs/evidence/radiance_mx_vpu_80f84ca_upstream_repro_20261010/README.md)
also reproduces the FP8 and derived FP4 four-tile MX+VPU programs byte for
byte on Nicolas's Spike extension, with 65,536 matching outputs in each case.
The [connected two-matmul MX+VPU chain](docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/README.md)
also runs from current upstream model2MLIR `e9ded36`; its objects, ELF, and
Spike log match the earlier frontend run byte for byte.
The [128³ plain MX chain](docs/evidence/nicolas_connected_plain_chain_128_266c593/index.json)
uses the [current model2MLIR capture](docs/evidence/nicolas_plain_chain_128_model2mlir_e9ded36/index.json)
and compiler-issued MM1→resident MM2 commands on Nicolas's stock Spike.
All 16,384 FP8 codes and 512 scales match at each site. The independent
[MM2 check](docs/evidence/nicolas_resident_mm2_128_266c593/index.json)
starts from source C1 data to isolate the resident second contraction.
A [source-derived 64×128×128 row-prefix chain](docs/evidence/nicolas_connected_plain_chain_64x128_d512fc2/index.json)
also compiles and runs on the same Spike model, matching 8,192 FP8 codes and
256 scales at each site from two independent builds.
The [16-row prefix ladder](docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/index.json)
extends this source-backed connected lowering through every M from 16 to 128
in steps of 16, with full C1/C2 Spike comparisons and independent replays.
A [linkable RV64 object path](docs/evidence/nicolas_resident_pair_object_1eebfc5/index.json)
compiles the typed connected graph into a data-free `mx_issue.o`; M = 16, 96,
and 128 objects pass the same full-output Spike comparisons.
A [fresh published checkout](docs/evidence/nicolas_connected_plain_chain_128_fresh_checkout_ae945d0/index.json)
rebuilds the native verifier and reproduces the connected program and Spike log.
The [row-major BF16 readout](docs/compiled_mx_pipeline.md#direct-row-major-bf16-readout-across-output-tiles)
writes four output tiles directly into one logical matrix. Pinned Spike matches
65,536 FP4 and FP8 outputs with VPU ×2 and 16,384 FP8 outputs retiled to
64×64; FP4 and FP8 also emit data-free objects with a row-major output ABI.
One FP4 object is independently [rebound twice on Spike](docs/compiled_mx_pipeline.md#direct-row-major-bf16-readout-across-output-tiles)
to distinct source-derived payloads and matches 131,072 BF16 outputs without
host output rearrangement.
The [Muon MMIO object path](docs/compiled_mx_pipeline.md#muon-mmio-issuer-handoff)
emits data-free RV32 issuers for selected FP8+VPU and FP4+VPU physical
programs, with gateway busy waits. [Source-bound Muon ELFs](docs/compiled_mx_pipeline.md#source-bound-muon-kernel-and-simulator-gap)
link checked operands and full BF16 verifiers. The stock Cyclotron co-model
lacks the compiler's move-in, move-out, and VPU commands. An isolated model
patch executes each selected precision with 0 / 65,536 BF16 mismatches in
two independent builds per precision. FP4 uses a generated Radiance fixture
and tile-major output; this is functional-model evidence, with RTL and FPGA
qualification still open.
The [four-tile FP4 variant](docs/compiled_mx_pipeline.md#generated-four-tile-fp4-gemm-with-tilewise-vpu-epilogue)
also matches 65,536 BF16 outputs on pinned Spike. It uses Radiance's pinned
generator and golden model to create a new 256×256 FP4 fixture; Radiance has
no committed driver for this case, so this result does not claim source ELF parity.
The [captured scalar addition](docs/compiled_mx_pipeline.md#captured-bf16-scalar-adds-across-four-mx-output-tiles)
now lowers `matmul + 1.5` through four VPU ADDS commands for FP8 and generated
FP4. Two independent Spike builds per precision match all 65,536 derived BF16
outputs. The general source CLI also reproduces the FP8 capture-driven program
and result twice.
An [ordered captured affine epilogue](docs/compiled_mx_pipeline.md#ordered-scalar-mx-vpu-chain-from-a-pytorch-capture)
now issues `matmul * 2.0 + 1.5` as eight VPU commands over four tiles.
Two Spike builds each for FP8 and generated FP4 match all 65,536 BF16 outputs,
with BF16 rounding after each scalar operation.
The [Cyclotron source check](docs/compiled_mx_pipeline.md#executed-generated-fp4-source-tiles-on-cyclotron)
finds that the derived 256×256 Muon driver builds but computes only one output
tile. Four separately built source-derived 128×128 tiles reassemble to the
full generated golden; their BF16 ×2 reference matches the compiler's Spike
test. The source check uses a pinned functional Cyclotron model, not RTL.
The [runtime object check](docs/compiled_mx_pipeline.md#reusable-four-tile-fp4-mxvpu-object-on-spike)
issues the same data-free MX+VPU object twice on Nicolas's pinned Spike model
with distinct source-derived FP4 inputs and matches all 131,072 BF16 outputs.
The [committed-source FP8 object check](docs/compiled_mx_pipeline.md#reusable-committed-source-fp8-object-on-spike)
also rebinds one Rocket object to two 128×128×512 FP8 payloads and matches all
32,768 BF16 outputs on the pinned Spike model.
The [FP6 object check](docs/compiled_mx_pipeline.md#reusable-two-wave-fp6-object-with-runtime-luts-on-spike)
rebinds packed codes, scales, and all A/B/C LUT banks for a generated
128×128×1024 source fixture. Its serial two-wave schedule matches 32,768 BF16
outputs on the pinned Spike model.
The [batched decode projections](docs/compiled_mx_pipeline.md#batched-decode-gemv-projections)
match all source BF16 outputs for FP8 batches 32, 64, and 128 and FP4 batch
128, including the two non-square output tiles.
The [GQA QK audit](docs/compiled_mx_pipeline.md#gqa-qk-numerical-domain-and-source-derived-candidate)
finds that the current attention generator's Q/K bytes overflow Nicolas's
reduced-precision MX path. An explicitly requantized first-tile candidate
matches 4,096 BF16 outputs on Spike; unchanged attention source parity remains
open.
The [linkable QK object](docs/compiled_mx_pipeline.md#linkable-qk-object-with-runtime-buffers)
uses runtime operand pointers. One compiler-emitted RoCC object matches two
different source-derived QK payloads on Nicolas's Spike, with 8,192 BF16
outputs checked. The Muon-produced P handoff and mixed attention execution
remain separate qualification gates.
The [first PV proxy](docs/compiled_mx_pipeline.md#first-pv-contraction-with-a-muon-p-proxy)
is captured from PyTorch through model2MLIR and lowered from typed MX MLIR.
It matches all 4,096 BF16 outputs on Spike using source-derived V and P encoded
with a Torch-exp proxy for Muon's exponential unit. An
[executed Cyclotron cross-check](docs/compiled_mx_pipeline.md#executed-muon-to-mx-pv-cross-check-on-cyclotron)
now shows the source Muon kernel produces the same 4,096 P codes and 128
scales, and its first MX PV tile matches all 4,096 Spike BF16 outputs. The
full [16-tile PV roster](docs/compiled_mx_pipeline.md#all-16-executed-gqa-pv-cutpoints-on-nicolas-spike)
also matches 65,536 BF16 outputs through one compiler emitted MX object on
Nicolas Spike, after an isolated Cyclotron overwrite-model correction. The
source [final recurrence](docs/compiled_mx_pipeline.md#executed-final-gqa-recurrence)
also reconstructs all 32,768 BF16 O values from executed PV tiles and Muon
row state. The full mixed compiler path and RTL FPEX parity remain open.
The same stream also
executes an in-place BF16 ×2 VPU epilogue for FP8 and FP4 with exact derived
goldens. Typed FP8, FP4, and FP6 quantized readouts match the pinned Spike
convention for codes and scales. The FP6 path uses the checked-in source header
for its output specialization; its 16-wave schedule selects FP6 output only on
the final K wave.
The source-bound full
MM1→VPU×2→SPAD_REQUANT→resident MM2 chain now runs in one compiler-generated
RV64 program and matches the C1/C2 source outputs on Spike. The checked
frontend and target inputs now form one SSA-connected typed chain for the
qualified 64³ E4M3 source case; general mixed lowering remains open.
The [BF16 VPU softmax seam](docs/compiled_mx_pipeline.md#compiler-issued-bf16-vpu-softmax)
captures `torch.softmax` with model2MLIR, issues configuration, transfers, six
VPU operations, and readout, and matches all 512 BF16 source-reference outputs
on Spike.
Nicolas's [full VPU source oracle](docs/compiled_mx_pipeline.md#nicolas-vpu-source-oracle-across-all-operations)
also passes all 29 checks on pinned Spike, including fused EXPSUB/EXPSUM;
the [base](docs/compiled_mx_pipeline.md#compiler-issued-base-vpu-operations),
[variant](docs/compiled_mx_pipeline.md#compiler-issued-vpu-broadcast-and-same-bank-variants),
and [ordering](docs/compiled_mx_pipeline.md#compiler-issued-vpu-dependencies-and-memory-ordering)
receipts now map each source check to an independently compiled RoCC program
with full BF16 comparison. General scheduling and RTL/FPGA validation remain
separate.
The [fused VPU compiler path](docs/compiled_mx_pipeline.md#compiler-issued-fused-expsub-and-expsum)
now lowers model2MLIR `e9ded36` captures of EXPSUB and EXPSUM to executable
commands; Nicolas's Spike matches all 512 outputs for each and all 128 EXPSUM
sum values.
The two legal E4M3×FP4 modes in Nicolas's standalone asymmetric profile also
lower through the shared physical command representation and match all 4,096
BF16 source outputs each on pinned Spike. The selected MX+VPU profile has no
asymmetric compute mode; see
[asymmetric qualification](docs/compiled_mx_pipeline.md#nicolass-standalone-asymmetric-mode).
Nicolas's separate FP6 E3M2 LUT × FP4 direct profile also runs through that
physical path and matches all 4,096 BF16 source outputs on pinned Spike.
The reverse FP4 direct × FP6 E3M2 LUT profile does as well, with its own
source-bound mode and 4,096 / 4,096 BF16 Spike match.
The separate E4M3 LUT × E2M3 LUT profile also passes all 4,096 source BF16
outputs, including the weight alternate-format command bit.
Nicolas's E5M2 LUT × FP4 direct profile passes the same full-output Spike
check and exercises the activation alternate-format bit.
Direct E4M3 activation × E3M2 LUT weight and FP4 activation × direct E4M3
weight also pass; these cases qualify the full-width activation and weight
transfer layouts, respectively.
The complete dedicated DIM16 asymmetric source matrix now passes 26 of 26
legal modes across 20 Rocket profiles, with 106,496 source BF16 outputs per
matrix run and an independent full reproduction. The
[matrix qualification](docs/compiled_mx_pipeline.md#dedicated-dim16-asymmetric-mode-matrix)
records the exact scope and pinned receipts.
The DIM8 and DIM32 all-asymmetric Rocket profiles each pass all 21 checked-in
source modes on Nicolas's Spike, with independent reproductions. Nicolas's
pinned generator supplies the other 15 mode tests per mesh. Fourteen of those
pass twice, bringing stock Spike coverage to **35 / 36 legal modes** on each
mesh. Direct E4M3 activation × E4M3 LUT weights fails on the stock model; the
same ELF passes with an isolated weight-LUT model correction. The
[mesh matrix receipts](docs/compiled_mx_pipeline.md#dim8-and-dim32-all-asymmetric-source-matrices)
keep that diagnostic separate from stock qualification.
The separate DIM16 all-asymmetric Rocket profile passes 30 source modes:
26 asymmetric pairs, three same-format LUT tests, and direct FP4×FP4. The
29-mode subset has two full reproductions; the 30-mode suite ran once and
FP4×FP4 has an independent reproduction. Its 36 legal modes leave six
without checked-in Nicolas source tests; see the
[DIM16 all-asymmetric receipts](docs/compiled_mx_pipeline.md#dim16-all-asymmetric-source-matrix).
The Radiance 64×64×64 direct E4M3×E4M3 fullout driver adds a separate
31st legal mode: a generated Radiance source header, model2MLIR capture,
physical command stream, and two Spike builds match all 4,096 BF16 outputs.
Its [receipt](docs/compiled_mx_pipeline.md#radiance-direct-fp8-on-the-all-asymmetric-profile)
keeps that source provenance distinct from Nicolas's checked-in matrix.
Nicolas's pinned `gen_asym.py` produces independent headers for the five
remaining DIM16 cells. Four additional modes match every BF16 output twice
on the stock Spike model, bringing distinct numerical coverage to **35 / 36**.
Direct E4M3 activation × E4M3 LUT weights is the remaining gap: stock Spike
reports 4,094 mismatches, while a local model correction matching the RTL's
weight-LUT lane selection reports zero for the same ELF. The
[generated-mode evidence](docs/compiled_mx_pipeline.md#generated-dim16-mode-probes)
keeps that patch diagnostic separate from stock-model qualification.
An [isolated patched-Spike rerun](docs/compiled_mx_pipeline.md#isolated-weight-lut-spike-correction-across-all-legal-modes)
now matches all 36 legal modes on DIM8, DIM16, and DIM32: 105 stock-passing
ELFs retain their output logs, and the three stock failures pass with the
same ELFs. Stock Spike remains at 35 / 36 per mesh; RTL and FPGA qualification
for the patch remain open.
The source-bound scheduler also passes Nicolas's larger direct E4M3×FP4
128×128×128 DIM16 and 128×128×256 DIM32 kernels on Spike, comparing all
16,384 BF16 outputs in each. Their [generated programs and reproduction
receipts](docs/compiled_mx_pipeline.md#larger-asymmetric-source-shapes)
record the current shape scope. The three other checked-in 128×128×256
asymmetric DIM32 kernels also pass with 64-line LUT banks. The matrix CLI now
discovers and qualifies all five larger source cases by shape.
The asymmetric CLI now binds hashes and physical layouts of Nicolas's packed
operand, scale, LUT, and golden arrays to the typed contraction before
lowering; a fresh E4M3×E2M3 Spike run reproduces the result with that binding.
The physical `WaitIdle` primitive polls the Muon gateway busy register at offset `0x20`;
the standalone Rocket path has no proven equivalent completion endpoint and
refuses that primitive.

The existing quantization contract and numeric diagnostics are scoped to
`GemminiMxFPConfigs.standaloneMxFPConfig` at Gemmini
`f0167390b56fb315deea90ac1fc3983772e92d82` and MxGen
`a27ce3cd81513210c21f971ec3977defd13fa21e`. It carries an explicitly
selectable Merlin support provider with an empty executable capability claim.
It has no Merlin runtime backend, matrix lowering, oracle, or certified fact
bundle. Its layout rules are source-derived; narrow payload-to-RTL
diagnostics are described below. Other shapes and general command scheduling
remain unreviewed. Separate structural profiles now cover 41 Gemmini fragments
and 40 Chipyard wrappers at Gemmini `266c593` and MxGen `dba3e7e`; those profiles
do not inherit the older contract's numerical qualification.

The [quantization handoff](docs/quantization_handoff.md) defines the selected
software contract, explicit per-site policy, TorchAO adapter, model2MLIR
manifest, and MX MLIR dialect. The handoff is a checked
operation plan. It does not execute an accelerator contraction or certify a
whole model. The source files below retain their narrow layout and simulator
diagnostics.
The [derivation record](docs/derivation_status.md) distinguishes source-checked
facts, authored policy, handwritten algorithms, and Phase 1/2 work. Merlin's
`examples/mx_gemmini/` owns the experiment recipe and target metadata; this
repository owns the OOT contract, RTL gate, dialect, and handoff. The reusable
TorchAO operand capture code lives in `microscaling-quant`.
The [Phase 0 review](docs/phase0_review.md) records the numerical scope checked
and the inputs still needed before a new Phase 0 run can be admitted.
The [latest-RTL diagnostic](docs/latest_rtl_diagnostic.md) records the separate
`2029218` candidate build and scale-load checks; it does not change the active
Merlin provider contract. The OOT contract gate accepts that exact candidate
for explicit operand-only capture, with its `unreviewed` status preserved.
The [candidate iteration roster](docs/iteration_roster_candidate.md) gives
four independent MX-sized capture models and an artifact materializer.

`profiles/mx-gemmini-rocket-2029218.json` identifies the standalone
`MxGemminiRocketConfig` candidate and binds its two Chipyard source files by
SHA-256. The historical 64×64×64 FP8 capture receipt and handoff are under
`docs/evidence/model2mlir_radiance_mx_gemm_20261006.*`. The PyTorch inputs
are not the handwritten FP8 code and scale blobs; this capture verifies
frontend structure and operation selection, not numerical output parity or
executable MX command lowering.
The [October 7 recapture](docs/evidence/model2mlir_radiance_mx_gemm_20261007.json)
used the newest locally fetched model2MLIR `main` revision `7915e23` and the
current MX issuer commit `0153e66`. It produced the same source MLIR and
handoff SHA-256 values as the October 6 receipt, with no opaque calls. The
GitHub head could not be refreshed in this environment, so `7915e23` is a
local checkout identity rather than a claim about the live remote head.

`tests/capture_radiance_mx_gemm.py` now reads a selected source driver and
data header and captures that driver's dimensions through the supplied
model2MLIR checkout. Its default is the feasible 128×128×512 FP8 GEMM with
128-wide K tiles. With `--profile` and `--rtl-root`, it binds the typed MX
handoff to the selected current RTL profile and records both source and target
scratchpad placement in the receipt. `tools/match_source_gemm.py` audits the
entire source shape ladder. See [source kernel matching](docs/source_kernel_matching.md).
The October 9 source-bound capture uses model2MLIR revision `7485a829`.
The selected 256 KiB MX profile requires different scratchpad and scale-buffer
addresses from the handwritten 128 KiB library. The earlier source-data C
diagnostics matched the source goldens; the compiler-generated command path
independently matches them for FP8, FP4, and FP6.
The generated source FP4 64×64×128 driver follows the same capture and
profile binding path. Its two-wave diagnostic matched all 4,096 BF16 golden
outputs on the pinned Spike extension. The checked-in FP6 128×128×2048 source
now binds all row-specific LUTs and packed operands to a typed MX contract;
its serial diagnostic matched all 16,384 BF16 golden outputs. The FP6 frontend
policy captures structure with only the source's first LUT line, while the
source payload supplies the actual per-row data. The pinned Spike LUT path
needs a documented serial scale-buffer workaround. A
[two-wave FP6 experiment](docs/compiled_mx_pipeline.md#alternating-fp6-scale-halves-on-nicolass-spike)
now runs the compiler's alternating command schedule on an isolated corrected
Spike model: the same ELF has 16,362 BF16 mismatches on pinned Spike and zero
on the corrected model. A [scoped RTL module probe](docs/evidence/nicolas_scale_mem_rtl_266c593/README.md)
checks Nicolas's actual scale-memory half selectors at DIM16; full RTL and
FPGA execution remain unqualified.
The same [alternating schedule](docs/compiled_mx_pipeline.md#alternating-fp6-scale-halves-on-nicolass-spike)
also reproduces all 16,384 BF16 outputs of the checked-in Radiance
128×128×2048 FP6 driver across 16 K waves on that isolated Spike model.

Verify the standalone source binding with
`python3 tools/check_profile.py --profile profiles/mx-gemmini-rocket-2029218.json
--chipyard /path/to/chipyard`. This checks the selected config chain and
refuses drift in either hashed Scala source.

`mx_gemmini_support.layout` implements:

- `CONFIG_EX` activation/weight format codes;
- DIM16 logical scale-row ordering and format-specific physical bank address
  decoding;
- first-buffer E8M0 scale payloads and funct-27 `rs2`;
- capacity-bounded K-wave partitioning and zero-based scale payload slicing;
- conversion of logical E8M0 scale matrices from activation [M][K/32] and
  weight [N][K/32] or [K/32][N] into those wave payloads;
- `CONFIG_SCALE_MEM` `rs1` bitfield encoding;
- FP6 E3M2 codebook bit packing, 8-byte DMA padding, and funct-29 `rs2`;
- exact FP6 code-to-index mapping for supplied 16-entry codebooks, plus the
  activation and weight nibble layouts used by the selected DIM16 loop;
- direct FP8 byte layout and direct FP4 nibble layout for logical A[M][K]
  and B[K][N] operands;
- a contraction payload builder that checks operand and scale shapes together
  and slices both into matching K waves for all three formats.

Each planned wave fits the first 4 KiB active window for both operands and
the 9-bit `CONFIG_SCALE_MEM` K bound. This older builder is a layout/capacity
plan; the separate `physical_program` lowering schedules the source-bound BF16
subset. The pinned `ScaleFactorMem` counters wrap after a complete configured
loop; rs1 bit 62 resets requantizer counters but is not wired to the scale read
counters. RTL evidence for alternating FP6 scales remains outstanding.

The older packer refuses payloads above one active window. The source-bound
physical lowering issues separate scale uploads per K wave and has executed a
2048-wide FP6 contraction on Spike. BF16 multi-output tiling is qualified for
one FP8 shape; quantized readout is qualified for selected single-output FP8,
FP4, and FP6 shapes. Further shapes and legal mode classes remain open.
`tools/check_radiance_header.py` compares a caller-selected source FP8 data
header with this OOT planner. For the checked-in Radiance
`m128n128k512` header, every A/B code and E8M0 scale byte matches exactly,
and the selected U250 profile admits the planned funct-27 uploads. The
[receipt](docs/evidence/radiance_mx_fp8_m128n128k512_payload_20261007.json)
records source/profile hashes, per-wave payload hashes, and command fields.
This is analytical byte parity; it does not issue the contraction or compare
the source hardware golden output.
The FP6 transform currently accepts one configured spatial window of at most
128 rows and columns, in multiples of 32. It requires every element code to
appear in its assigned codebook line. Choosing codebooks or approximating
missing codes is a numerical policy outside this compiler layout prototype.
Larger spatial dimensions require further tiling and LUT reload scheduling.
The selected RTL's `MX_LOAD_LUT` DMA reads through the next 8-byte boundary,
so callers must use the padded result as the actual source buffer.

Reproduce a pinned diagnostic C source under an explicit artifact root:

```sh
python -m mx_gemmini_support.bringup --format mxfp6 --case split32x64 \
  --output /configured/artifact-root/split32x64-mxfp6.c
```

The available cases are `square32`, `square64`, and `split32x64` for each of
`mxfp8`, `mxfp6`, and `mxfp4`. The command prints the C source SHA-256 and
refuses to overwrite an existing output. Compile it with the pinned Gemmini
headers and bare-metal support using `MX_ROCKET`; the selected source-bound
simulator, not this generator, supplies the hardware verdict.

This package's `linear_contraction_operands` returns the logical code and E8M0
scale tensors in the builder's expected orientations for a rank-2 Linear.
`mx_gemmini_support.model2mlir.plan_linear_operands` checks the byte tensor
types and packs them. FP6 requires caller-supplied exact codebooks. With
TorchAO installed, run
`python -m pytest tests/test_model2mlir.py -q`: the integration test applies
the actual TorchAO transform to one 32x32x32 Linear per format with two
operand cases: distinct row/column magnitudes and an all-zero block. It
verifies that the resulting six C sources have the same SHA-256 digests as
the programs executed on the source-bound RTL simulator. All six tested
programs exited zero with no BF16 mismatches. The zero blocks produced E8M0
code 104. The test does not rerun the simulator; these remain narrow
diagnostics, not whole-model or general lowering.
`iter_spatial_tiles` slices rank-2 to rank-4 logical operands into aligned
M/N windows of at most 128 rows and columns without materializing a packed
whole matrix. Its indexed results retain batch axes and output tile offsets;
each tile carries the matching A rows, B columns, E8M0 scales, and, for FP6,
the corresponding exact LUT lines from caller-supplied global codebooks.
The iterator prepares payloads but does not assemble output tiles, choose
FP6 codebooks, or schedule accelerator transfers.
`plan_independent_batches` also slices model2MLIR's visible functional matmul
handoff along matching rank-3 or rank-4 batch axes and returns one indexed
payload per rank-2 contraction. Its common FP6 codebook must contain every
selected code in every batch. This prepares buffers; it does not schedule a
batched attention kernel or qualify the host softmax seam.
The bounded `emit_independent_batches_baremetal_c` composes up to four matching
single-window payloads as successive independent contractions. A two-batch
rank-3 functional matmul source is pinned for each format; the exact ELFs
exited zero with both BF16 matrices matching on source-built Spike. The exact
FP8 batch ELF also exited zero on the selected RTL simulator: both independent
BF16 matrices matched. Earlier shorter attempts hit wall-time or cycle
bounds before comparison. FP6 and FP4 batches still have Spike evidence only.
These serial contractions do not establish an attention kernel or its host
softmax seam.
`emit_spatial_tiles_baremetal_c` serially checks a complete 32x32 tile grid
of one rank-2 contraction with output dimensions up to 64x64 and K=32.
The quantizer handoff supplies a 64x32 activation and 32x64 weight with
distinct output quadrants. Its four-tile C source is hash-pinned in the
integration test for each format. All twelve tile results across MXFP8,
MXFP6, and MXFP4 matched BF16 expectations on the source-built Spike core.
This diagnostic neither assembles a combined output buffer nor establishes
an RTL verdict or a general tiled runtime schedule.

A separate 32x32x32 functional-matmul vector exercises one representable
element subnormal per output for each of MXFP8, MXFP6, and MXFP4. The test
pins the emitted C source hashes, including exact FP6 LUT contents. The same
three compiled programs exited zero with no BF16 mismatches on both the
selected RTL simulator and a source-built Spike core with the selected
libgemmini extension. Their local diagnostic receipt is
`subnormal-source-bound-crosscheck-20260929.json` under the configured
artifact root. This checks three particular vectors, not all underflow,
rounding, or nonfinite behavior.

Separate source-bound RTL diagnostics ran FP8, FP4, and FP6 32x32x32
contractions using this package's packed activation, weight, and E8M0
buffers. FP6 also used its exact code-to-index mapping and packed LUT
buffers. Each used distinct values in the upper and lower activation rows
and left and right weight columns. All four BF16 output quadrants matched
exactly for each format, and each simulator process exited zero. The command
sequences were initially hand-written for these diagnostics.
`mx_gemmini_support.diagnostic_program.emit_single_window_baremetal_c` now
renders a bounded C command sequence from the coherent payload and a caller's
BF16 expectation matrix. Its emitted 32x32x32 and 64x64x64 programs passed
the same source-bound RTL simulator with zero BF16 mismatches for all three
formats. The 64-case FP6 program uploaded 32 LUT lines per operand. The
emitter refuses other shapes, multiple K waves, or FP6 LUT granularity other
than shift one. This is a reusable bringup program for two square one-window
shapes, not an executable Merlin provider or a general DMA/loop schedule.

For a separate K-wave diagnostic, `max_blocks_per_wave=1` makes the coherent
payload builder split a 32x32x64 contraction into two 32-element K waves.
`emit_two_wave_baremetal_c` reloads both scale banks and operand tiles for
the second wave, sets the RTL `ex_accumulate` bit, and keeps the FP6 LUT
resident. One source-bound RTL run per format matched all BF16 values for
distinct first- and second-wave activation codes. The emitter's compiled
load images are byte-identical to those tested programs for FP8, FP6, and
FP4. This qualifies only that two-wave shape and command sequence; larger
capacity-driven wave plans still need simulator checks.

For the full test suite, point Python at the pinned model2MLIR `e9ded36` and
MXQuant `b4af543` checkouts, then run from this directory:

```sh
PYTHONPATH="$MODEL2MLIR_ROOT:$MXQUANT_ROOT" python -m pytest tests -q
```

This also prevents a different editable model2MLIR installation in the active
virtualenv from supplying an older capture API.
The [GitHub Actions contract gate](.github/workflows/contract-evidence.yml)
checks issuer fields, illegal profile modes, and the archived 31-driver and
108-mode receipts from a fresh Python install. A second job checks all 81
profiles against a fresh checkout of Nicolas's pinned Gemmini and MxGen
sources. Rebuilding the numerical Spike runs still requires the pinned
Radiance, model2MLIR, and RV64 toolchains documented in
[the pipeline](docs/compiled_mx_pipeline.md).

To inspect the provider through Merlin, set `MERLIN_TARGET_PATH` to this
directory and resolve `mx_gemmini`. This selects its metadata only; no
compiler execution is enabled by selection.
