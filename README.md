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
`physical_program.lower_bound_source` schedules configuration, scale/LUT DMA,
operand movement, K-wave compute, and BF16 or supported quantized readout for
complete output tiles.
`tools.qualify_source_mx` builds a standalone RV64 ELF and compares every
output on Nicolas's pinned Spike extension. Qualified source shapes span the
FP8 and FP4 ladder; see
[compiled source parity](docs/compiled_mx_pipeline.md). The same stream also
executes an in-place BF16 ×2 VPU epilogue for FP8 and FP4 with exact derived
goldens. Typed FP8, FP4, and FP6 quantized readouts also match Nicolas's current
Spike convention for codes and scales. The FP6 path uses the checked-in fullout
header as an explicit quantized output specialization; its 16-wave schedule
selects FP6 output only on the final K wave. Source header output goldens use
older or different conventions, so
[quantized source parity remains open](docs/compiled_mx_pipeline.md#fp8-and-fp4-quantized-readout).
The source-bound full
MM1→VPU×2→SPAD_REQUANT→resident MM2 chain now runs in one compiler-generated
RV64 program and matches the C1/C2 source outputs on Spike. The two frontend
and target MLIR inputs are checked and joined by site ID and source hashes;
one connected SSA-level chain and general mixed lowering remain open.
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
source modes on Nicolas's Spike, with independent reproductions. Each profile
has 36 legal modes, so 15 modes per profile still need source tests and numerical
qualification; see the [mesh matrix receipts](docs/compiled_mx_pipeline.md#dim8-and-dim32-all-asymmetric-source-matrices).
The separate DIM16 all-asymmetric Rocket profile passes all 26 available source
modes in two fresh runs. Its 36 legal modes leave 10 without source tests;
see the [DIM16 all-asymmetric receipts](docs/compiled_mx_pipeline.md#dim16-all-asymmetric-source-matrix).
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
needs a documented serial scale-buffer workaround.

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

Run `python -m pytest tests -q` from this directory.

To inspect the provider through Merlin, set `MERLIN_TARGET_PATH` to this
directory and resolve `mx_gemmini`. This selects its metadata only; no
compiler execution is enabled by selection.
