# MX Gemmini software contract and MLIR handoff

Licensed under Apache-2.0; see [LICENSE](LICENSE).

The MLIR dialect includes `readout_to_smem` and `wait` operations for
an explicit MX-to-Muon shared-memory handoff. The `radiance-mlir` composition
package verifies their ordering with Muon fences and barriers against a
selected SoC profile. Matrix contraction and handoff operations are contract IR.
The new physical `vpu_execute` and `spad_requant` operations lower to Rocket
RoCC commands for source-bound DIM16 MX+VPU profiles; see
[current MX profiles and VPU lowering](docs/current_mx_profiles.md).
`mx_gemmini_support.command_ir` represents checked physical command fields
and emits the same stream through Rocket RoCC or the Muon-side Radiance MMIO
gateway. `transfer_ir.plan_uploads` binds scale and optional LUT uploads to a
selected MX profile and materializes one command list per K wave. Operand
tile movement, compute scheduling, readout, and numerical qualification are
still required before the MLIR contraction is executable. The physical
`WaitIdle` primitive polls the Muon gateway busy register at offset `0x20`;
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
the 9-bit `CONFIG_SCALE_MEM` K bound. This is a layout/capacity plan only:
the caller still must order scale uploads, configure the wave's loop bounds,
execute the corresponding operand tiles, and preserve the BF16 accumulator
across K waves. The pinned `ScaleFactorMem` counters wrap after a complete
configured loop; rs1 bit 62 resets requantizer counters but is not wired to
the scale read counters. Correct command scheduling and K-wave arithmetic
still require matching RTL simulator evidence. No larger contraction is
claimed executable by this prototype.

The packer refuses payloads above one active window. A compiler must
schedule additional uploads and choose scale banks for larger contractions;
that scheduling and all MX arithmetic lowering remain open.
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
