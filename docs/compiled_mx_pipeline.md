# MX Gemmini compiler path and qualification

This repository targets Nicolas's `gemmini-mx-cleanup` revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. The profile exporter covers
its 41 MX fragments and 40 Chipyard wrappers. The first executable lowering
uses `MxE4M3Fp4VpuGemminiRocketConfig` for FP8 and FP4, and
`MxE3M2OnlyGemminiRocketConfig` for FP6. The former has two 8-lane BF16 VPUs,
the fused `EXPSUB`/`EXPSUM` operations, and `SPAD_REQUANT`. Nicolas's current
branch has no single profile combining that VPU with FP6 E3M2 LUT compute.

## Compiler inputs and output

`tests/capture_radiance_mx_gemm.py` captures the selected PyTorch shape through
model2MLIR `7485a829c0195af0ec42820837d609e62e466564`, then
`bind_profile.py` selects an RTL-derived legal PE mode. The capture specifies
the contraction structure. Source numerical data is an **explicit
specialization**: `source_payload.py` reads the Radiance header's packed A/B
bytes, E8M0 scales, BF16 golden, and all three 64-line FP6 LUT banks into
hashed binary resources. `bind_payload.py` attaches their manifest digest and
origin to the typed `mx_gemmini.contract`. It does not claim that the captured
PyTorch tensor generated those source bytes.
For an FP8 or FP4 source driver with quantized output, the same binding specializes
the terminal BF16 readout into typed `mx_gemmini.readout_quantized`, returning
packed codes and E8M0 scales. The capture receipt identifies this output
specialization separately from the PyTorch matmul capture.

`physical_program.py` verifies the MLIR, manifest, and profile together. It
plans scratchpad placement and lowers complete BF16 output tiles into ordered Rocket
commands: configuration, LUT and E8M0 DMA, operand mvin, scale selection,
K-wave loop compute, and BF16 mvout. `command_ir.py` checks pointer fields and
emits the physical RoCC instruction stream. `standalone.py` embeds the bundle
as binary data in an assembly object and emits a generic full-output checker.
The generated program never includes the handwritten source data header or a
shape-specific replacement C kernel.

From this repository, run one command after the source driver and adjacent
data header are available:

```sh
python -m tools.qualify_source_mx \
  --mlir docs/evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir \
  --driver /path/to/radiance-kernels/kernels/gemm_mxgemmini/mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /new/output-directory
```

The command validates the source/profile, writes `bundle/`,
`payload_bound.mlir`, `build/physical_program.json`, `build/mx_issue.c`,
`build/mx_data.S`, `build/mx_program.elf`, and
`build/artifact_manifest.json`, then runs the ELF against the selected RTL
checkout's pinned `libgemmini` Spike extension. It refuses an existing output
directory and rejects profile, resource, MLIR, or gitlink drift. The artifact
manifest records source, profile, tool, program, object, ELF, extension, and
Spike-log digests.

For FP4, the Radiance tree generates the 64×64×128 data header on demand.
The exact source revision and generation command are in
[source kernel matching](source_kernel_matching.md#generated-source-fp4-kernel).
The 64×64×128 FP8 and 128×128×128 FP4 runs likewise use generated headers
and fresh [FP8](evidence/model2mlir_radiance_mx_fp8_64x64x128_capture_receipt.json)
and [FP4](evidence/model2mlir_radiance_mx_fp4_128x128x128_capture_receipt.json)
model2MLIR captures.
The 128×128×256 FP8 and FP4 headers are generated at the same source revision
with `python kernels/gemm_mxgemmini/gen_mxgemm_data.py fp8 128 128 256` and
the corresponding `fp4` command after building `lib/golden/mx_golden`.
Their [FP8 capture](evidence/model2mlir_radiance_mx_fp8_128x128x256_capture_receipt.json)
and [FP4 capture](evidence/model2mlir_radiance_mx_fp4_128x128x256_capture_receipt.json)
record the generator, model2MLIR, policy, source, and profile identities.
The 256×256×256 FP8 header uses the same generator; its
[capture](evidence/model2mlir_radiance_mx_fp8_256x256x256_capture_receipt.json)
records that the original 128 KiB source scratchpad cannot place the C tile
beside double-buffered operands, while the selected Nicolas MX+VPU profile's
256 KiB scratchpad can. Its numerical reference comes from the source golden
generator, not execution of the original driver at that size.

## Full-output Spike evidence

| Precision and source tile | MX profile | BF16 outputs matched | Receipt |
|---|---|---:|---|
| FP8 128×128×512, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp8_128x128x512_tk128_20261009.json) |
| FP8 128×128×512, K tile 256 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp8_128x128x512_tk256_20261009.json) |
| FP8 64×64×128, K tile 64 | MX+VPU E4M3/FP4 | 4,096 | [receipt](evidence/compiled_mx_fp8_64x64x128_20261009.json) |
| FP8 64×64×64, K tile 64 | MX+VPU E4M3/FP4 | 4,096 | [capture](evidence/model2mlir_radiance_mx_fp8_64x64x64_tk64_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_64x64x64_tk64_fullout_20261009.json) |
| FP8 64×64×512, K tile 64, MX stage of SIMT contention | MX+VPU E4M3/FP4 | 4,096 | [capture](evidence/model2mlir_radiance_mx_fp8_64x64x512_tk64_simt_contention_mx_stage_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_64x64x512_tk64_simt_contention_mx_stage_20261009.json) |
| FP8 128×128×256, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp8_128x128x256_20261009.json) |
| FP8 128×128×128, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x128_tk128_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x128_tk128_fullout_20261009.json) |
| FP8 128×128×256, K tile 256 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x256_tk256_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x256_tk256_fullout_20261009.json) |
| FP8 256×256×256, four 128×128 output tiles | MX+VPU E4M3/FP4 | 65,536 | [receipt](evidence/compiled_mx_fp8_256x256x256_20261009.json) |
| FP8 128×128×2048, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x2048_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x2048_20261009.json) |
| FP8 128×128×5632, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x5632_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x5632_20261009.json) |
| FP4 64×64×128, K tile 64 | MX+VPU E4M3/FP4 | 4,096 | [receipt](evidence/compiled_mx_fp4_64x64x128_20261009.json) |
| FP4 64×64×64, K tile 64 | MX+VPU E4M3/FP4 | 4,096 | [capture](evidence/model2mlir_radiance_mx_fp4_64x64x64_tk64_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_64x64x64_tk64_fullout_20261009.json) |
| FP4 128×128×128, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp4_128x128x128_20261009.json) |
| FP4 128×128×256, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp4_128x128x256_20261009.json) |
| FP4 128×128×512, K tile 512 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x512_tk512_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x512_tk512_fullout_20261009.json) |
| FP4 128×128×1024, K tile 512 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x1024_tk512_fullout_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x1024_tk512_fullout_20261009.json) |
| FP4 128×128×2048, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x2048_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x2048_20261009.json) |
| FP4 128×128×5632, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x5632_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x5632_20261009.json) |
| FP6 128×128×2048, K tile 128 | E3M2 LUT, no VPU | 16,384 | [receipt](evidence/compiled_mx_fp6_128x128x2048_20261009.json) |

Each receipt records a zero-mismatch pinned Spike run. Independent output
directories reproduced identical payload-bound MLIR, physical source files,
objects, ELF, extension, and Spike log hashes for FP8 128×128×512 with both
K tile sizes and FP4 64×64×128.
The FP8 K tile 256 run tests a distinct schedule over the same source data;
the 256-deep runs use fresh PyTorch/model2MLIR captures and generated source
headers.
The 2048- and 5632-deep FP8/FP4 runs likewise use fresh captures and headers
generated at Radiance revision `94ba7ca8afe213b92fa2428689fa64800c8eeca9`
with `gen_mxgemm_data.py <fp8|fp4> 128 128 <2048|5632>`. Each Spike run
matches all 16,384 BF16 values from its generated header. The capture receipt
records the generator and header hashes, while the payload-bound MLIR and
Spike receipt bind those bytes to the physical command stream. These checks
cover long K schedules of 16 and 44 waves; they do not imply the handwritten
source C driver was itself run on this Spike build.
The 64×64×64 FP8 and FP4 single-tile fullout drivers also match their
generated BF16 goldens. The `mxgemm.simt_contention.cpp` capture checks its
64×64×512 **MX GEMM stage** against the source golden; the driver's Muon warp
traffic and MX↔SIMT contention timing are outside this standalone Rocket
Spike test and remain a Radiance integration gate.
The 128-deep FP8 and 512/1024-deep FP4 headers were generated at the same
source revision. Their respective K tiles of 128, 512, and 512 all match the
source BF16 goldens. The FP8 256-deep K tile 256 run also matches, using the
selected 256 KiB MX+VPU scratchpad where the original 128 KiB source layout
cannot place that C tile beside double-buffered A/B tiles.
For FP6, the selected Spike LUT path ignores the alternating scale selector;
the compiler's explicitly named `spike_serial` mode reloads scale half zero.
`rtl_alternating` plans the target's intended halves but is not a numerical
qualification for RTL or FPGA.

## FP8 and FP4 quantized readout

The compiler now emits quantized FP8 and packed FP4 readout commands from a source-bound
model2MLIR contraction. The payload contains both the Radiance header's
`C_out`/`C_scales_row` and a separately named Nicolas reference computed from
the source BF16 golden. FP8 uses the current MXQuant power-of-two scale,
hardware 2^-23 floor, and round-to-nearest-even E4M3 grid. FP4 uses the
current output scale (code zero for an all-zero block), BF16→E3M1→E2M1
projection, and nibble packing along M.
The references are independent of Spike. FP8's 16,384 codes and 512 scales,
and FP4's 2,048 packed bytes and 128 scales, matched the pinned MXQuant
implementation exactly in the
[FP8 oracle check](../tests/test_quantized_lowering.py) and
[FP4 oracle check](../tests/test_source_quantized_payload.py). The standalone
checker compares both codes and scales on Nicolas's pinned Spike.

| Source shape and tile | Spike vs Nicolas reference | Radiance header difference | Evidence |
|---|---:|---|---|
| FP8 64×64×64, one 64×64 tile | 4,096 codes + 128 scales exact | 4,091 codes + 128 scales | [capture](evidence/model2mlir_radiance_mx_fp8_64x64x64_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_64x64x64_quant_20261009.json) |
| FP8 128×128×256, one 128×128 tile | 16,384 codes + 512 scales exact | 16,376 codes + 512 scales | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x256_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x256_quant_20261009.json) |
| FP4 64×64×64, one 64×64 tile | 2,048 packed bytes + 128 scales exact | Header codes are FP8; 128 scales differ | [capture](evidence/model2mlir_radiance_mx_fp4_64x64x64_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_64x64x64_quant_20261009.json) |
| FP4 128×128×128, one 128×128 tile | 8,192 packed bytes + 512 scales exact | Header codes are FP8; 512 scales differ | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x128_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x128_quant_20261009.json) |
| FP8 128×128×128, K tile 128 | 16,384 codes + 512 scales exact | 16,370 codes + 512 scales differ | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x128_tk128_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x128_tk128_quant_20261009.json) |
| FP8 128×128×256, K tile 256 | 16,384 codes + 512 scales exact | 16,376 codes + 512 scales differ | [capture](evidence/model2mlir_radiance_mx_fp8_128x128x256_tk256_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp8_128x128x256_tk256_quant_20261009.json) |
| FP4 128×128×512, K tile 512 | 8,192 packed bytes + 512 scales exact | Header codes are FP8; 512 scales differ | [capture](evidence/model2mlir_radiance_mx_fp4_128x128x512_tk512_quant_capture_receipt.json), [Spike](evidence/compiled_mx_fp4_128x128x512_tk512_quant_20261009.json) |

The discrepancy is an upstream convention change, not a passing source-golden
test: `radiance-kernels/lib/golden/mx_golden.cpp` uses `log2_pmax = 8` for FP8
requant output, whereas Nicolas's current
`src/main/scala/gemmini/MxRequantizer.scala` and pinned Spike use
`log2_pmax_floor = 0`. The generated Radiance codes and scales therefore
cannot match this MX+VPU configuration until that source golden is updated or
the selected hardware convention changes. The 128×128×256 source driver also
cannot stage its C tile in the original 128 KiB scratchpad; its generated
golden is still usable, and Nicolas's 256 KiB profile admits the tile.
The Radiance generator's `C_out` currently uses FP8 output even for generated
FP4 headers. The compiler qualifies FP4 against the separately named current
reference and records this source format mismatch.

## FP6 LUT-indexed quantized readout on Nicolas's Spike

The checked-in FP6 128×128×2048 header contains packed operands, three LUT
banks, BF16 `C_out_bf16`, projected `C_proj_hw`, and output scales. The
matching checked-in driver is **fullout**, while the FP6 requant drivers in
Radiance refer to 128 and 512-deep headers that are absent from this source
revision. The `--fp6-quantized-specialization` option explicitly reuses the
PyTorch/model2MLIR fullout contraction and source operands and changes its
terminal readout to typed `mx_gemmini.readout_quantized` with FP6 E3M2 LUT
indices. It does not claim to have captured the absent FP6 requant header.

For a 16-wave K loop, the physical lowering keeps intermediate outputs in
BF16, then selects FP6 output on the final wave. The initial attempt to select
FP6 on every wave failed Spike because requantization overwrote intermediate
accumulator values. The final program now matches an independent BF16→E3M2→C
LUT reference on Nicolas's pinned Spike: **8,192 packed index bytes and 512
E8M0 scales, all exact**. The header's `C_proj_hw` differs in 8,175 bytes and
its 512 output scales all differ from the current target convention. This is
target-convention parity, with the source discrepancy recorded explicitly in
the [qualification](evidence/model2mlir_radiance_mx_fp6_128x128x2048_quant_qualification.json)
and [Spike receipt](evidence/compiled_mx_fp6_128x128x2048_quant_20261009.json).
An independent output directory reproduced the same payload-bound MLIR,
generated source files, objects, ELF, extension, and Spike log hashes; its
[second receipt](evidence/compiled_mx_fp6_128x128x2048_quant_repro_20261009.json)
records the check. The linker diagnostic log includes the output directory
path, so that log's hash differs.
The FP6 output path uses the E3M2-only profile; Nicolas's current MX+VPU
profile has no E3M2 LUT compute mode.

Reproduce it with the same profile, RTL, toolchain, and fullout driver as the
BF16 row above, adding `--fp6-quantized-specialization` to
`python -m tools.qualify_source_mx`. The command emits the
[payload-bound MLIR](evidence/model2mlir_radiance_mx_fp6_128x128x2048_quant_payload_bound.mlir)
and a standalone RV64 ELF, then checks both codes and scales on Spike.

The same lowering accepts ordered physical `mx_gemmini.vpu_execute` and
`mx_gemmini.spad_requant` operations between the contraction and BF16 readout.
Two checked [FP8](../examples/mx_fp8_vpu_x2_profile_bound.mlir) and
[FP4](../examples/mx_fp4_vpu_x2_profile_bound.mlir) programs add an in-place
VPU `MULS` by BF16 2.0. The expected outputs are derived independently by
exact BF16 exponent shifts from the source BF16 golden; the checker refuses
values for which that derivation is invalid. On Nicolas's Spike extension,
both compiler-generated mixed command streams matched every derived output:
[FP8 receipt](evidence/compiled_mx_fp8_vpu_x2_20261009.json) and
[FP4 receipt](evidence/compiled_mx_fp4_vpu_x2_20261009.json). These epilogues
were explicitly added to the model2MLIR contraction captures; they are a
compiler composition test, not a claim that the Radiance source GEMM contains
those VPU operations. Other VPU or SPAD_REQUANT operations lower to physical
commands but the standalone source-golden checker rejects them until their
result golden is bound.

### Compiler-generated matrix→VPU→requant program

The FP8 64×64×128 Radiance fullout driver has a fresh
PyTorch→model2MLIR contraction capture and generated source header. The
`--vpu-spad-requant-x2` specialization binds those exact source operand bytes,
then appends typed `mx_gemmini.vpu_execute` and
`mx_gemmini.spad_requant` operations. The physical program computes both
K waves in BF16, multiplies the complete BF16 C tile by 2.0 on the VPU, and
requantizes it to tiled FP8 codes with resident E8M0 scales. The BF16 C
scratchpad rows are `[256,768)` and the packed destination is `[1024,1280)`;
the latter reuses space only after the input matrix operands are dead. The
SPAD_REQUANT scale destination is a runtime pointer whose 33-bit field is
checked before issue.

An independent exact BF16 exponent shift followed by the current FP8 output
reference supplies the expected codes and scales. The compiler-generated
RV64 ELF matched **4,096 FP8 codes and 128 E8M0 scales** on Nicolas's pinned
Spike. Separate output directories reproduced the generated source, objects,
ELF, extension, and Spike log hashes. See the [typed MLIR](evidence/matrix_vpu_requant_fp8_64x64x128_payload_bound.mlir),
[qualification](evidence/matrix_vpu_requant_fp8_64x64x128_qualification.json),
[Spike receipt](evidence/compiled_matrix_vpu_requant_fp8_64x64x128_20261009.json),
and [second receipt](evidence/compiled_matrix_vpu_requant_fp8_64x64x128_repro_20261009.json).
The source header's unscaled quantized output differs in 4,094 codes and
all 128 scales; it is retained as a separate source reference. This is an
explicit target composition test, not a claim that the handwritten Radiance
GEMM contains VPU operations. The Nicolas source chain below qualifies the
resident second matrix operand separately and then in a single program with
the first matrix.

Reproduce with the normal `tools.qualify_source_mx` command above, selecting
the FP8 64×64×128 fullout driver and its matching model2MLIR bound capture,
and adding `--vpu-spad-requant-x2`.

Nicolas's reference `vpu_ops`, `vpu_softmax`, and
`chain_vpu_spad_requant` programs were also built and run directly against
the pinned Spike extension. They passed all reference comparisons, including
the fused VPU cases and the VPU→requant→matmul chain. That is model capability
evidence; the [reference receipt](evidence/nicolas_vpu_spike_reference_20261009.json)
records source, ELF, tool, and log hashes. The dialect's physical
VPU/SPAD_REQUANT command lowerer now has a source-audited executable seam:
`source_vector_chain.py` checks the exact `VPU_MULS` and tiled, resident
`SPAD_REQUANT` calls in Nicolas's `chain_vpu_spad_requant.c`, then emits
[typed MLIR](evidence/nicolas_chain_vpu_requant_64x64_source_bound.mlir).
The compiler binds the E8M0 scale destination as a checked runtime pointer,
issues generic BF16 input and tiled FP8 output transfers around those typed
commands, and builds a standalone RV64 ELF. Its independent BF16×2→FP8
reference equals the source header's 4,096 C1 codes and 128 source scales
incremented by one. The compiler ELF matches all 4,096 codes and 128 scales
on pinned Spike ([receipt](evidence/compiled_nicolas_chain_vpu_requant_64x64_20261009.json));
a second output directory reproduced the MLIR, generated sources, objects,
ELF, extension, and Spike log hashes
([second receipt](evidence/compiled_nicolas_chain_vpu_requant_64x64_repro_20261009.json)).
This qualification begins with the source C1 BF16 tile and ends at the C1
requantized tile. The x2 epilogues above establish one compiler-generated
matrix/VPU composition. The following stages extend it through MM2 and MM1.

The `--with-resident-matmul` option now continues that same typed seam through
`mx_gemmini.resident_contract`. The operation names the resident C1 tile, B2
source buffers, C2 destination, and E4M3 format. The profile verifier checks
the 64×64×64 DIM16 geometry, scratchpad lifetimes, and output mode; the
physical lowerer emits B2 scale and weight transfers, a resident output-scale
configuration, and `LOOP_WS_SPAD` without reloading C1 or its scales. Its
source audit checks Nicolas's B2 transfer and second-matmul calls, and an
independent exact BF16×2 output oracle matches his C2 header. The generated
ELF compares **4,096 C1 codes, 128 C1 scales, 4,096 C2 codes, and 128 C2
scales**; all match on pinned Spike. Two clean output directories reproduce
the typed MLIR, generated source, objects, ELF, extension, and Spike log
hashes. See the [typed MLIR](evidence/nicolas_chain_vpu_requant_resident_64x64_source_bound.mlir),
[Spike receipt](evidence/compiled_nicolas_resident_chain_64x64_20261009.json),
and [reproduction](evidence/compiled_nicolas_resident_chain_64x64_repro_20261009.json).
This path starts from Nicolas's C1 BF16 source tile, so it still excludes MM1.

The new `tools.capture_nicolas_chain` command captures
`torch.matmul(torch.matmul(A, B1), B2)` with model2MLIR
`7485a829c0195af0ec42820837d609e62e466564`. It selects two distinct
64×64×64 MX FP8 sites, produces a profile-bound handoff accepted by
`mx-gemmini-opt`, and reports no opaque frontend operations. Independent
captures reproduce the source MLIR, handoff, bound MLIR, and quantization
manifest hashes ([receipt](evidence/model2mlir_nicolas_chain_two_site_capture_20261009.json),
[reproduction](evidence/model2mlir_nicolas_chain_two_site_capture_repro_20261009.json),
[bound MLIR](evidence/model2mlir_nicolas_chain_two_site_profile_bound_20261009.mlir)).
The PyTorch examples supply contraction structure; Nicolas's header supplies
the actual packed bytes and scales.

With both frontend artifacts, the compiler now emits **one ordered RV64
program** for MM1→VPU×2→SPAD_REQUANT→resident MM2. MM1 loads Nicolas's
A1/B1 codes and scales and writes BF16 C1 to a live scratchpad tile; VPU and
requant consume it there; MM2 consumes tiled C1 codes and resident activation
scales without a DRAM reload. A diagnostic mvout copies MM1's BF16 C1 for
comparison while the chain continues to use the on-chip tile. The program
compares all **4,096 C1 BF16 values, 4,096 C1 FP8 codes, 128 C1 scales, 4,096
C2 FP8 codes, and 128 C2 scales** against Nicolas's source header and its
independent exact ×2 oracle. Every comparison matches on the pinned Spike
extension. Two independent builds reproduce the typed MLIR, generated
source, objects, ELF, extension, and Spike log hashes. See the
[source-bound typed chain](evidence/nicolas_full_chain_source_bound_20261009.mlir),
[Spike receipt](evidence/compiled_nicolas_full_chain_20261009.json), and
[reproduction](evidence/compiled_nicolas_full_chain_repro_20261009.json).
This is the qualified 64×64×64 E4M3 MX+VPU mode; the two-site frontend MLIR
and the typed source specialization are separate checked compiler inputs.
They are bound by site IDs and hashed source/profile receipts, and the
physical lowering produces one program. A connected SSA-level chain and
generalization across shapes and mode classes remain to be implemented.

Reproduce the seam with:

```sh
python -m tools.qualify_nicolas_vector_requant \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/output-directory
```

Add `--with-resident-matmul` to this command to reproduce the second-matmul
program. To reproduce the full chain, also supply
`--frontend-bound-mlir docs/evidence/model2mlir_nicolas_chain_two_site_profile_bound_20261009.mlir`
and
`--frontend-receipt docs/evidence/model2mlir_nicolas_chain_two_site_capture_20261009.json`.
The output directory must be new.

## Nicolas's standalone asymmetric mode

`MxAsymE4M3Fp4GemminiRocketConfig` admits E4M3 activation through the
4-bit-addressed, 8-bit-entry LUT with direct FP4 weights (`pe_mode=10`), and
direct 8-bit E4M3 activation with direct FP4 weights (`pe_mode=6`).
`MxE4M3Fp4VpuGemminiRocketConfig` admits only symmetric FP8 and symmetric
FP4; it cannot execute this asymmetric mode. A target profile is therefore
selected before emitting the command sequence.

`tools.qualify_nicolas_asym` captures a fresh 64×64×64 PyTorch matmul through
the pinned model2MLIR checkout, then requires an explicit recipe naming
Nicolas's `matmul_tiled_asym_e4m3_fp4_64x64.c`, its packed data header, the
frontend site ID, the selected profile digest, and the exact legal compute
tuple. The ordinary symmetric handoff binder continues to reject this mode.
The specialized MLIR is accepted by `mx-gemmini-opt`; the source-bound
lowerer emits LUT, scale, operand DMA, compute, and BF16 readback commands
into a standalone RV64 ELF. On Nicolas's pinned Spike extension, the ELF
matches **all 4,096 BF16 source golden values**. Two fresh captures and builds
match the model2MLIR artifacts, generated issue source, RV64 objects and ELF,
Spike extension, and Spike log hashes. The
[receipt](evidence/compiled_nicolas_asym_e4m3_fp4_20261009.json) and
[independent reproduction](evidence/compiled_nicolas_asym_e4m3_fp4_repro_20261009.json),
[specialized MLIR](evidence/model2mlir_nicolas_asym_e4m3_fp4_bound_20261009.mlir),
and [generated issue source](evidence/compiled_nicolas_asym_e4m3_fp4_issue_20261009.c)
record the command and provenance checks. The random PyTorch examples
establish graph structure and shape; Nicolas's header supplies the physical
packed operands and scales. This is a bounded source specialization for the
standalone asymmetric config, not a claim that model2MLIR currently emits
mixed-operand quantization or that the MX+VPU profile supports it.

The same command with `--variant direct` selects Nicolas's separate
`matmul_tiled_asym_e4m3s_fp4_64x64.c` and data header. This changes the
activation projection, physical PE mode, operand layout, and number of M
tiles. Its generated ELF also matches **all 4,096 BF16 source values** on
pinned Spike. Two fresh builds reproduce the captured frontend artifacts,
generated source, objects, ELF, extension, and Spike log hashes. See the
[direct-mode receipt](evidence/compiled_nicolas_asym_direct_e4m3_fp4_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_direct_e4m3_fp4_repro_20261009.json),
[bound MLIR](evidence/model2mlir_nicolas_asym_direct_e4m3_fp4_bound_20261009.mlir),
and [generated issue source](evidence/compiled_nicolas_asym_direct_e4m3_fp4_issue_20261009.c).
The source PyTorch capture and unbound handoff hashes equal those in the LUT
mode evidence; the recipes and bound contracts differ.

The default qualification command now lowers both modes into the same
`PhysicalProgram` representation used by the symmetric MX compiler. The
lowerer exports Nicolas's packed arrays as hashed binary resources, checks
the typed site and profile mode, and emits ordered configuration, LUT or
disable, scale, DMA, compute, and readout commands. The existing standalone
emitter produces a generic RoCC command issuer, data object, and BF16
checker; it does not copy either source C kernel into the executable.

| Activation path | PE mode | Physical commands | Spike BF16 matches | Evidence |
|---|---:|---:|---:|---|
| E4M3 via LUT × FP4 direct | 10 | 63 | 4,096 / 4,096 | [receipt](evidence/compiled_nicolas_asym_physical_lut_20261009.json), [reproduction](evidence/compiled_nicolas_asym_physical_lut_repro_20261009.json), [command stream](evidence/compiled_nicolas_asym_physical_lut_program_20261009.json) |
| E4M3 direct × FP4 direct | 6 | 69 | 4,096 / 4,096 | [receipt](evidence/compiled_nicolas_asym_physical_direct_20261009.json), [reproduction](evidence/compiled_nicolas_asym_physical_direct_repro_20261009.json), [command stream](evidence/compiled_nicolas_asym_physical_direct_program_20261009.json) |

The [LUT issuer](evidence/compiled_nicolas_asym_physical_lut_issue_20261009.c)
and [direct issuer](evidence/compiled_nicolas_asym_physical_direct_issue_20261009.c)
are generated from those physical streams. Each pair of fresh builds matches
the capture, resources, command program, issue source, objects, ELF,
extension, and Spike log hashes. The earlier bounded C diagnostics remain
available with `--diagnostic` for their archived receipts; the command below
uses the physical stream by default.

Reproduce with a new output directory:

```sh
python -m tools.qualify_nicolas_asym \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAsymE4M3Fp4GemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/output-directory
```

Use `--variant direct` for mode 6. The default `lut` selects mode 10.

### FP6 E3M2 activation with FP4 weights

Nicolas's `MxAsymFp6Fp4GemminiRocketConfig` has a separate legal mode:
FP6 E3M2 activation through a 6-bit-entry LUT, direct FP4 weights, and PE
mode 3. `--variant fp6_lut` selects his
`matmul_tiled_asym_fp6_fp4_64x64.c` and checked-in data header. The compiler
exports the 32 activation, weight, and output LUT lines as 384-byte resources
each, binds their source hash to typed MLIR, and lowers the contraction to
63 physical commands. The generated standalone ELF matches **all 4,096 BF16
source values** on pinned Spike. Two independent builds reproduce the
frontend artifacts, resources, physical program, issuer, objects, ELF,
extension, and Spike log hashes: [receipt](evidence/compiled_nicolas_asym_physical_fp6_fp4_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_physical_fp6_fp4_repro_20261009.json),
[bound MLIR](evidence/model2mlir_nicolas_asym_fp6_fp4_bound_20261009.mlir),
and [physical program](evidence/compiled_nicolas_asym_physical_fp6_fp4_program_20261009.json).

The current frontend policy captures an FP8 matmul site as a structural
starting point. The explicit source recipe changes that site's activation to
FP6 E3M2 LUT and its weight to FP4 direct; model2MLIR has not yet generated
this mixed quantization or the source's row-specific LUTs from PyTorch data.
This qualification belongs to the standalone FP6×FP4 profile; the selected
MX+VPU profile does not contain the mode.

```sh
python -m tools.qualify_nicolas_asym --variant fp6_lut \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAsymFp6Fp4GemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/output-directory
```

### FP4 activation with FP6 E3M2 weights

Nicolas's `MxAsymFp4Fp6GemminiRocketConfig` reverses the mixed pair:
direct FP4 activation, FP6 E3M2 weights through a 6-bit-entry LUT, and PE
mode 1. `--variant fp4_fp6_lut` binds
`matmul_tiled_asym_fp4_fp6_64x64.c` and its checked-in data header to a fresh
64×64×64 model2MLIR matmul capture. The source recipe explicitly specializes
the structural FP8 handoff to this mixed operand contract; it does not imply
that model2MLIR produced the packed FP4/FP6 values. The shared physical
lowerer emits 63 commands, including the weight LUT loads. The standalone
ELF matches **all 4,096 BF16 source values** on the pinned Spike extension.
Two independent builds match the captured frontend artifacts, resources,
physical program, generated issuer, objects, ELF, extension, and Spike log
hashes: [receipt](evidence/compiled_nicolas_asym_physical_fp4_fp6_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_physical_fp4_fp6_repro_20261009.json),
[bound MLIR](evidence/model2mlir_nicolas_asym_fp4_fp6_bound_20261009.mlir),
[source recipe](evidence/model2mlir_nicolas_asym_fp4_fp6_recipe_20261009.json),
and [physical program](evidence/compiled_nicolas_asym_physical_fp4_fp6_program_20261009.json).
This mode belongs to the standalone FP4×FP6 profile; the selected MX+VPU
profile has no asymmetric mode.

```sh
python -m tools.qualify_nicolas_asym --variant fp4_fp6_lut \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAsymFp4Fp6GemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/output-directory
```

## Remaining gates

1. Reconcile the FP8/FP4 source requant goldens with Nicolas's current convention,
   then qualify the actual FP6 requant source drivers once their missing data
   headers are available, the other asymmetric legal mode classes, and any
   source shapes without receipts. Extend multi-output tiling beyond the
   qualified FP8 BF16 shape.
2. Consolidate the two checked MLIR inputs into one connected chain, then
   generalize its explicit scratchpad lifetimes beyond the qualified 64³
   Nicolas source case.
3. Qualify every legal mode class on the matching Spike/RTL configuration,
   and keep unsupported profile combinations rejected. FP6+VPU requires a
   new RTL configuration and profile before it can be advertised.
4. Validate the alternating FP6 scale path against RTL, then qualify the
   Radiance MMIO/FPGA issue path separately from Rocket RoCC.
