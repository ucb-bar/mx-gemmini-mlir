# MX Gemmini layout prototype

This out-of-tree compiler prototype is scoped to
`GemminiMxFPConfigs.standaloneMxFPConfig` at Gemmini
`f0167390b56fb315deea90ac1fc3983772e92d82` and MxGen
`a27ce3cd81513210c21f971ec3977defd13fa21e`. It carries an explicitly
selectable Merlin support provider with an empty executable capability claim.
It has no Merlin runtime backend, matrix lowering, oracle, or certified fact
bundle. Its layout rules are source-derived; narrow payload-to-RTL
diagnostics are described below. Other shapes and general command scheduling
remain unreviewed.

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

model2MLIR's `linear_contraction_operands` returns the logical code and E8M0
scale tensors in this builder's expected orientations for a rank-2 Linear.
`mx_gemmini_support.model2mlir.plan_linear_operands` consumes that handoff,
checks its byte tensor types, and packs it without changing model2MLIR or
TorchAO. FP6 requires caller-supplied exact codebooks. With the model2MLIR
branch on `PYTHONPATH` and TorchAO installed, run
`python -m pytest tests/test_model2mlir.py -q`: the integration test applies
the actual TorchAO transform to one 32x32x32 Linear per format with two
operand cases: distinct row/column magnitudes and an all-zero block. It
verifies that the resulting six C sources have the same SHA-256 digests as
the programs executed on the source-bound RTL simulator. All six tested
programs exited zero with no BF16 mismatches. The zero blocks produced E8M0
code 104. The test does not rerun the simulator; these remain narrow
diagnostics, not whole-model or general lowering.
`plan_independent_batches` also slices model2MLIR's visible functional matmul
handoff along matching rank-3 or rank-4 batch axes and returns one indexed
payload per rank-2 contraction. Its common FP6 codebook must contain every
selected code in every batch. This prepares buffers; it does not schedule a
batched attention kernel or qualify the host softmax seam.

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
