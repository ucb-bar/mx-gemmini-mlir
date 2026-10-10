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
origin to the typed `mx_gemmini.contract`. New bound modules also carry the
canonical manifest in `mx.payload_manifest_json`: resource names, shapes,
layouts, byte counts, and hashes are inspectable in MLIR. The Python verifier
checks the manifest structure, per-row FP6 LUT banks, and digest; `mx-opt`
checks its SHA-256 against the contract binding. Physical lowering compares
that manifest with the source bundle before using its external byte arrays.
Current bindings use `mx.payload_binding_schema = "source_resources_ssa_v1"`:
typed `mx_gemmini.resource` results replace the captured contraction's four
code/scale inputs, and `mx_gemmini.upload_lut` consumes each available LUT
resource before the contraction. The source-specialized function drops its
now-unused captured tensor arguments. The verifier checks names, tensor shapes,
hashes, and upload order. Earlier archived MLIR uses the digest-only binding;
its physical stream remains reproducible through the pinned source bundle.
The capture does not claim that the PyTorch tensor generated those source bytes.
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

The FP6 case was rebuilt twice from clean compiler `c1bf581` with the
canonical resource manifest embedded in
[bound MLIR](evidence/fp6_resource_manifest_binding_266c593/payload_bound.mlir).
Its [resource manifest](evidence/fp6_resource_manifest_binding_266c593/payload_manifest.json)
names the packed A/B arrays, E8M0 scales, and all three 64×3 FP6 LUT banks.
The [physical commands](evidence/fp6_resource_manifest_binding_266c593/physical_program.json)
and [generated RoCC issuer](evidence/fp6_resource_manifest_binding_266c593/mx_issue.c)
produce **16,384 / 16,384 BF16 matches** on the pinned Spike extension. The
[qualification index](evidence/fp6_resource_manifest_binding_266c593/qualification.json)
links both build receipts and the [Spike log](evidence/fp6_resource_manifest_binding_266c593/spike.log).
The two builds reproduce the MLIR, physical program, issuer, ELF, and Spike
log hashes after excluding the path-bearing link log hash. Their 1,289-command
physical stream has the same hash as the prior FP6 result. This run uses the
serial Spike schedule with its recorded FP6 scale-selector workaround; it does
not qualify the alternating-buffer schedule on RTL or FPGA.

Compiler `0d6460c` adds SSA source resources. Its
[six-case qualification index](evidence/ssa_source_binding_266c593/qualification.json)
links two independent builds each for
[FP6 BF16](evidence/ssa_source_binding_266c593/fp6/bound.mlir),
[FP4 BF16](evidence/ssa_source_binding_266c593/fp4/bound.mlir),
[FP8 matrix→VPU→requant](evidence/ssa_source_binding_266c593/fp8_vpu/bound.mlir),
Nicolas's [generated FP6 asymmetric mode](evidence/ssa_source_binding_266c593/asym_fp6_e3m2_e3m2/bound.mlir),
and FP4×E4M3 mixed modes on
[DIM8](evidence/ssa_source_binding_266c593/asym_dim8_fp4_e4m3/bound.mlir)
and [DIM32](evidence/ssa_source_binding_266c593/asym_dim32_fp4_e4m3/bound.mlir).
Their pinned stock Spike results match **16,384 FP6 BF16 outputs**, **4,096
FP4 BF16 outputs**, **4,096 FP8 codes plus 128 E8M0 scales** from the VPU
program, and **4,096 BF16 outputs in each asymmetric case**, respectively.
Each pair reproduces bound MLIR, physical commands, issuer, ELF, and Spike
log hashes after excluding only the path-bearing link log. All six physical
program hashes equal their previously qualified streams. The VPU result uses
Nicolas's current quantized oracle; the FP6 source run retains the serial
Spike scale-selector workaround. These are Rocket/Spike results and do not
qualify a Radiance FPGA image.

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
`rtl_alternating` plans the target's intended halves. An isolated corrected
Spike experiment now tests its two-wave FP6 schedule below; RTL and FPGA
qualification remain open.

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

The discrepancy is a convention difference: `radiance-kernels/lib/golden/mx_golden.cpp`
uses `log2_pmax = 8` for FP8 output, whereas Nicolas's current
`src/main/scala/gemmini/MxRequantizer.scala` and pinned Spike use
`log2_pmax_floor = 0`. The hardware requant path above retains Nicolas's
convention. The 128×128×256 source driver cannot stage its C tile in the
original 128 KiB scratchpad; Nicolas's 256 KiB profile admits the tile.
The Radiance generator's `C_out` uses FP8 output even for generated FP4
headers, so a source-compatible FP4-input run produces FP8 codes.

For exact source-header output, `--source-header-quantized` binds the
model2MLIR contraction to `mx_gemmini.readout_bf16` followed by the typed
`mx_gemmini.host_requantize` op with policy `radiance_header_fp8_v1`.
The physical lowering emits BF16 MX commands and a generated RV64 C epilogue
that implements the source header's scale and FP8 code rule. It uses neither
LLVM nor handwritten matrix commands. The compiler checks the derived codes
and scales against the source bundle before building the ELF; the standalone
driver compares the computed codes and scales on Spike. This is a compatibility
path with a host epilogue, so its performance is not hardware requant
performance. The separate hardware path remains available for resident MX
consumers. See the [source-header Spike evidence](evidence/radiance_header_requant_266c593/qualification.json)
for all six buildable FP8/FP4 quantized drivers in this source snapshot:
FP8 64×64×64, 128×128×128, and 128×128×256; FP4 input 64×64×64,
128×128×128, and 128×128×512. Every FP8 header code and E8M0 output scale
matched on the pinned Spike. The FP4-input source headers also specify FP8
output codes.

```sh
python -m tools.qualify_source_mx \
  --mlir docs/evidence/model2mlir_radiance_mx_fp8_64x64x64_quant_bound.mlir \
  --driver /path/to/radiance-kernels/kernels/gemm_mxgemmini/mxgemm.fp8.singletile.tm64tn64tk64.requant.cpp \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /new/radiance-header-fp8-run \
  --source-header-quantized
```

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

Reproduce the fullout hardware FP6 output path with the same profile, RTL,
toolchain, and fullout driver as the BF16 row above, adding
`--fp6-quantized-specialization` to `python -m tools.qualify_source_mx`.
The command emits the [payload-bound MLIR](evidence/model2mlir_radiance_mx_fp6_128x128x2048_quant_payload_bound.mlir)
and a standalone RV64 ELF, then checks both codes and scales on Spike.

The two FP6 requant drivers at 128×128×128 and 128×128×512 now have
source-derived headers. `tools.generate_radiance_fp6_header` slices the
checked-in 2048-K operands and A/B/C LUTs, then calls Radiance's `mx_golden`
for each depth. Before writing either header it recomputes the full 2048-K
BF16 result, both packed C layouts, and output scales and requires exact
agreement with the checked-in header. These generated headers are fixtures
for the missing source files; they are not claimed to be upstream Radiance
headers.

Fresh PyTorch→model2MLIR captures of both requant drivers bind their source
operands, E8M0 scales, and three 64-line LUT banks to typed MX MLIR. The
explicit `mx_gemmini.host_requantize` consumes the BF16 readout and C LUT.
The RV64 host epilogue reproduces the source BF16→FP6 projection and packs
the C LUT indices. On Nicolas's pinned Spike, both depths matched all
**8,192 packed source bytes and 512 E8M0 scales**. The separate MX hardware
FP6 readout also passed Nicolas's oracle for these depths; that output uses
a different scale and code convention. See the archived
[FP6 source qualification](evidence/radiance_fp6_requant_266c593/qualification.json),
which includes the generated headers, frontend captures, bound MLIR, payload
bundles, command issuers, Spike logs, and hashes. The selected profile is
`MxE3M2OnlyGemminiRocketConfig`; this is not an MX+VPU FP6 qualification.
The source capture reports a 128 KiB scratchpad assertion, while this selected
target profile exposes 256 KiB. Operand placement and output bytes were
checked on Spike; equal scratchpad capacity is not claimed.

```sh
python -m tools.qualify_source_mx \
  --mlir docs/evidence/radiance_fp6_requant_266c593/fp6_128x128x128/profile_bound.mlir \
  --driver /path/to/radiance-kernels/kernels/gemm_mxgemmini/mxgemm.fp6.singletile.tm128tn128tk128.requant.cpp \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup --riscv-root /path/to/riscv-tools \
  --out-dir /new/radiance-fp6-source-run --source-header-quantized \
  --generate-missing-fp6-header
# Use the matching 512 MLIR and driver filenames for the other source driver.
```

The last flag stages the fixture in the output directory, records its
generation receipt, and leaves the Radiance checkout untouched. The
Radiance checkout needs its checked-in 2048-K header and a built
`lib/golden/mx_golden`. The standalone generator remains available for
inspecting the fixture directly.

The sibling FP6 **fullout** drivers at 128×128×128, 128×128×256,
128×128×512, and 128×128×1024 use generated operand headers and separate
fresh PyTorch→model2MLIR captures. Their typed contractions lower to
standalone BF16 MX readout programs. All **16,384 BF16 values per driver**
matched the source golden on Nicolas's pinned Spike. The
[fullout qualification](evidence/radiance_fp6_fullout_266c593/qualification.json)
archives all four frontend captures, bound MLIR, packed payloads, physical
commands, generated issuers, receipts, and Spike logs. The 1024-depth case
also reproduced from a source tree without its generated header using the
one-command staged fixture path, with identical ELF and Spike log hashes.
These runs use the E3M2-only profile and the 256 KiB target scratchpad.

The [source parity roster](evidence/radiance_mx_gemm_source_parity_266c593.json)
ties all **31 `mxgemm.fp*.cpp` drivers** in this pinned Radiance snapshot to
source-output Spike receipts: 23 fullout drivers with exact BF16 comparisons
and eight requant drivers with exact source code and scale comparisons. Its
test rechecks every source hash, receipt hash, output count, simulator status,
and a frontend capture from model2MLIR `7485a829` for each driver. This
roster covers the precision-specific MX GEMM drivers on the
selected Rocket RoCC simulator profiles. The generic MX GEMM entry points,
other Radiance workloads, RTL timing, and FPGA execution remain separate
qualification gates.

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
`mx_gemmini.spad_requant` operations. The current typed function carries
BF16 and code/scale values along checked SSA edges through readout, VPU,
requant, and return. The physical program computes both
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
The [SSA-connected MLIR](evidence/matrix_vpu_requant_fp8_64x64x128_ssa_bound.mlir)
has two fresh [first](evidence/compiled_matrix_vpu_requant_fp8_64x64x128_ssa.json)
and [second](evidence/compiled_matrix_vpu_requant_fp8_64x64x128_ssa_repro.json)
Spike receipts. Its generated source, RV64 objects, ELF, and Spike log match
the earlier command-only MLIR path byte for byte.
The source header's unscaled quantized output differs in 4,094 codes and
all 128 scales; it is retained as a separate source reference. This is an
explicit target composition test, not a claim that the handwritten Radiance
GEMM contains VPU operations. The Nicolas source chain below qualifies the
resident second matrix operand separately and then in a single program with
the first matrix.

Reproduce with the normal `tools.qualify_source_mx` command above, selecting
the FP8 64×64×128 fullout driver and its matching model2MLIR bound capture,
and adding `--vpu-spad-requant-x2`.

The same SSA-connected specialization now derives its tensor types, output
tile placement, and requantized readout size from a checked single-tile FP8
source shape. Fresh PyTorch captures with model2MLIR `e9ded36` and Radiance
source `ee22e0b` qualified **64×64×64** (4,096 codes, 128 scales) and
**128×128×128** (16,384 codes, 512 scales) on the pinned MX+VPU Spike
extension. Both captures selected one contraction with zero opaque calls;
two independent captures and two RV64 builds per shape reproduced their
respective MLIR, object, ELF, extension, and Spike log hashes. The generated
source headers were rebuilt with Radiance's current data generator and match
the earlier headers byte for byte. See the [shape evidence index](evidence/fp8_vpu_requant_shapes_e9ded36/index.json).
This composition adds VPU×2 to the source GEMM; it does not claim that the
handwritten fullout GEMM uses VPU. Multiple output tiles and requant blocks
outside the selected hardware range remain rejected.

Nicolas's reference `vpu_ops`, `vpu_softmax`, and
`chain_vpu_spad_requant` programs were also built and run directly against
the pinned Spike extension. They passed all reference comparisons, including
the fused VPU cases and the VPU→requant→matmul chain. That is model capability
evidence; the [reference receipt](evidence/nicolas_vpu_spike_reference_20261009.json)
records source, ELF, tool, and log hashes. Compiler-issued VPU coverage now
includes the following source-audited cases.

### Nicolas VPU source oracle across all operations

The [source VPU receipt](evidence/nicolas_vpu_source_all_ops_266c593/receipt.json)
builds Nicolas's `bareMetalC/vpu_ops.c` with `VPU_FUSED=1` and runs it on his
pinned Spike extension. Its [log](evidence/nicolas_vpu_source_all_ops_266c593/spike.log)
passes **29 / 29** source-reference checks, including all 14 VPU opcodes,
EXPSUB and EXPSUM, broadcast/reduction forms, same-bank access, and dependent
VPU/memory ordering. Two clean builds produced identical ELF, extension, and
log hashes. Reproduce with:

```sh
python -m tools.qualify_nicolas_vpu_ops_source \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/nicolas-vpu-source
```

This source executable is the full VPU oracle. The compiler-issued base,
variant, and ordering qualifications below now cover all **29 named source
checks** on Nicolas's pinned Spike extension. Each case is compiled into an
independent ELF with the source input sequence and reference calculation;
the monolithic source benchmark remains a separate qualification.

### Compiler-issued base VPU operations

The [base VPU receipt](evidence/nicolas_vpu_elementwise_compiled_266c593/index.json)
captures 12 BF16 PyTorch operations through model2MLIR `e9ded36`: add,
subtract, multiply, maximum, scalar add and multiply, exp, reciprocal,
reciprocal square root, row maximum, row absolute maximum, and row sum. A
checked binding selects one typed `mx_gemmini.vpu_execute` per capture on
Nicolas's two-unit MX+VPU profile. The compiler emits the load/store
configuration, source transfers, funct-33 VPU command, and output transfers.
The C driver reproduces the exact `vpu_ops.c` input sequence, including the
unary special values, and computes its expected output with Nicolas's
`vpu_ref.h`. No accelerator command is handwritten in that driver.

All **4,992 / 4,992 BF16 outputs** across those 12 operations match on the
pinned Spike extension. Each elementwise operation checks 512 values; each
reduction checks 128 replicated-lane values. Two fresh runs with identical
compiler sources reproduced all 12 frontend captures, bound modules,
issuers, ELFs, and logs byte for byte
([reproducibility](evidence/nicolas_vpu_elementwise_compiled_266c593/reproducibility.json)).
The archived [ADD binding](evidence/nicolas_vpu_elementwise_compiled_266c593/add/bound.mlir)
and [RSUM binding](evidence/nicolas_vpu_elementwise_compiled_266c593/rsum/bound.mlir)
show elementwise and reduction examples. Reproduce with:

```sh
python -m tools.qualify_nicolas_vpu_elementwise \
  --model2mlir-root /path/to/model2MLIR \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/nicolas-vpu-base
```

For reductions, the source VPU reduces four scratchpad rows and all eight
lanes, then replicates the scalar across eight lanes. PyTorch identifies the
logical reduction while the binding records this physical layout and the
source BF16 rounding rule. These are Spike results, not RTL or FPGA results.

### Compiler-issued fused EXPSUB and EXPSUM

The [fused VPU receipt](evidence/nicolas_vpu_fused_compiled_266c593/index.json)
captures `exp(a - b)` and `exp(a - b)` plus a grouped sum from BF16 PyTorch
with model2MLIR `e9ded36`, then binds each graph to one typed VPU operation
on Nicolas's fused two-unit profile. The compiler emits all configuration,
input transfers, funct-33 compute, and output transfers. The C driver only
reproduces the source test's input generator and calls `vpu_ref.h` for the
BF16 golden. The [EXPSUB](evidence/nicolas_vpu_fused_compiled_266c593/expsub/bound.mlir)
program matches **512 / 512 BF16 outputs**; the
[EXPSUM](evidence/nicolas_vpu_fused_compiled_266c593/expsum/bound.mlir)
program matches **512 / 512 outputs and 128 / 128 grouped-sum values** on
pinned Spike. Two fresh builds reproduce both frontend captures, bound MLIR,
issuers, ELFs, extension, and output logs byte for byte.

```sh
python -m tools.qualify_nicolas_vpu_fused \
  --model2mlir-root /path/to/model2MLIR \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/nicolas-vpu-fused
```

PyTorch identifies the operation and shapes; the source VPU reference defines
the rounded BF16 results. These checks qualify the two fused opcodes on
Nicolas's Spike model. The base-op checks above cover the other 12 opcodes;
the source ordering variants and RTL/FPGA behavior remain separate gates.

### Compiler-issued VPU broadcast and same-bank variants

The [variant receipt](evidence/nicolas_vpu_variants_compiled_266c593/index.json)
captures Nicolas's remaining single-operation source forms through the same
model2MLIR revision. Eleven independent programs cover MUL and MAX with
same-bank or broadcast sources, broadcast SUB across two addresses, plain and
broadcast EXPSUB, both EXPSUM placements with their sum outputs, and RSUM
with reduction length one. The binding selects the actual source bank
addresses and transfers only the rows the VPU reads. The generated issuer
places no fence between DMA and VPU commands, matching the source ordering
test. Nicolas's input generator and `vpu_ref.h` are the driver oracle.

All **5,888 / 5,888 BF16 values** across these 11 programs match on pinned
Spike. Two clean runs reproduce their captures, bound MLIR, issuers, ELFs,
extension, and logs byte for byte
([reproducibility](evidence/nicolas_vpu_variants_compiled_266c593/reproducibility.json)).
The [same-bank MUL](evidence/nicolas_vpu_variants_compiled_266c593/mul_same_bank/bound.mlir)
and [broadcast EXPSUM](evidence/nicolas_vpu_variants_compiled_266c593/expsum_bcast/bound.mlir)
modules show the physical address and reduction bindings. Reproduce with:

```sh
python -m tools.qualify_nicolas_vpu_variants \
  --model2mlir-root /path/to/model2MLIR \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/nicolas-vpu-variants
```

These checks establish numerical and ordered-command parity on Nicolas's
Spike extension. The dependent-command cases are qualified separately below.

### Compiler-issued VPU dependencies and memory ordering

The [ordering receipt](evidence/nicolas_vpu_ordering_compiled_266c593/index.json)
contains three more model2MLIR captures and profile-bound typed command
modules. The chain issues ADD→in-place MULS→RMAX; the write-after-read case
issues ADD, then overwrites its source bank with DMA before reading the ADD
result; the two-unit case issues independent ADDS and EXP, then MUL reading
the ADDS result across VPUs. The compiler emits all transfers and VPU
commands with **no intervening fence**. The driver retains the exact source
input generator, a completion fence, and the `vpu_ref.h` oracle. The write-
after-read DMA is an explicit source-bound scheduling test: its side effect
is outside the pure PyTorch arithmetic graph.

The chain matches **128 / 128 BF16 values**, write-after-read **512 / 512**,
and the dual-VPU case **1,536 / 1,536**, including both independent outputs
and the dependent result. Two fresh builds reproduce every capture, bound
module, issuer, ELF, and Spike log byte for byte
([reproducibility](evidence/nicolas_vpu_ordering_compiled_266c593/reproducibility.json)).
The [chain module](evidence/nicolas_vpu_ordering_compiled_266c593/chain/bound.mlir)
and [dual module](evidence/nicolas_vpu_ordering_compiled_266c593/dual/bound.mlir)
show the command order. Reproduce with:

```sh
python -m tools.qualify_nicolas_vpu_ordering \
  --model2mlir-root /path/to/model2MLIR \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/nicolas-vpu-ordering
```

Together with the base and variant receipts, the coverage tests map every
one of Nicolas's 29 source log checks to a compiler-issued Spike comparison.
This is a source-specific schedule qualification. General graph scheduling,
the monolithic 29-case compiler executable, and RTL/FPGA validation are
separate work.

### Compiler-issued BF16 VPU softmax

`tools.qualify_nicolas_vpu_softmax` captures `torch.softmax` on a 16×32 BF16
tensor with pinned model2MLIR `e9ded36`. It checks the resulting max, subtract,
exp, sum, and division decomposition, then binds Nicolas's six-step BF16 VPU
schedule to the MX+VPU profile. The typed
[VPU MLIR](evidence/nicolas_vpu_softmax_full_ce54256/softmax.profile_bound.mlir)
lowers to a 17-command Rocket program: flush, load/store configuration, four
input transfers, six funct-33 VPU operations, and four output transfers. The
generated issuer replaces Nicolas's MX configuration, transfer, and VPU calls;
the source retains input generation, a completion fence, and its bit-exact VPU
reference checker. On
pinned Spike, all **512 BF16 outputs** match the reference. Two clean runs
reproduce the frontend MLIR, bound MLIR, generated issuer, objects, ELF,
extension, and Spike log hashes
([first](evidence/nicolas_vpu_softmax_full_ce54256/first.json),
[second](evidence/nicolas_vpu_softmax_full_ce54256/reproduction.json)).
The earlier [six-command compute-only qualification](evidence/nicolas_vpu_softmax_model2mlir_329718b/first.json)
is retained separately.
The source VPU rounds each intermediate to BF16; model2MLIR's PyTorch
decomposition uses FP32 intermediates. This qualifies the source VPU softmax
sequence against the source reference, not bit-exact PyTorch output.

```sh
python -m tools.qualify_nicolas_vpu_softmax \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-vpu-softmax
```

### Matrix/VPU/requant source chain

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
This is the qualified 64×64×64 E4M3 MX+VPU mode. The two-site frontend MLIR
and typed source specialization are checked compiler inputs. The connected
specialization combines them into one function with explicit SSA edges for
MM1→BF16 readout→VPU×2→resident requant→MM2. Its lowerer checks each edge,
the profile and source digests, and equality with the source-audited physical
commands. The generated ELF and Spike log match the earlier two-input build
byte for byte. See the [connected MLIR](evidence/nicolas_connected_chain_266c593.mlir),
[Spike receipt](evidence/compiled_nicolas_connected_chain_266c593.json), and
[reproduction](evidence/compiled_nicolas_connected_chain_266c593_repro.json).
This remains a source-bound 64³ specialization. General shape scheduling and
independent lowering of arbitrary `mx_gemmini.contract` graphs remain open.

The same typed `mx_gemmini.resident_contract` physical lowerer now accepts
Nicolas's **128³ FP8 resident MM2** under the plain
`MxGemminiRocketConfig`, which has a requantizer but no VPU or
`SPAD_REQUANT`. Its scratchpad validation derives the C1, B2, and C2 spans
from the operation's dimensions and rejects overlaps. The
[128³ source-bound MLIR and Spike archive](evidence/nicolas_resident_mm2_128_266c593/index.json)
preloads C1 codes and its transposed scales from Nicolas's checked-in chain
header, then lowers the SSA-bound MM2 op into B2 scale and weight transfers,
resident scale selection, compute, and C2 readout. The generated RV64 ELF
matches **16,384 C2 FP8 codes and 512 E8M0 scales** on stock Spike; a second
run reproduces the generated program, ELF, and log. This test starts at C1;
the following connected-chain check also compiles MM1.

The [128³ two-site frontend capture](evidence/nicolas_plain_chain_128_model2mlir_e9ded36/index.json)
uses current upstream model2MLIR `e9ded36`, the plain MX profile, and
`torch.matmul(torch.matmul(A, B1), B2)`. Both 128×128×128 contractions appear
as profile-bound `mx_gemmini.contract` sites with no opaque frontend calls.
Two independent captures reproduce the source MLIR, handoff, bound MLIR,
and manifest byte for byte. The PyTorch examples establish graph structure;
Nicolas's source header supplies the packed operands and numerical goldens.
The capture alone is frontend evidence; the connected lowering below binds
those two sites to the source payload.

```sh
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/mx-resident-mm2-128 \
  --baseline-manifest docs/evidence/nicolas_resident_mm2_128_266c593/artifact_manifest.json
```

Capture the two sites again with `python -m tools.capture_nicolas_chain
--matrix-dim 128` and the pinned model2MLIR, MXQuant, RTL, profile, and
`mx-gemmini-opt` paths; see the command's `--help` for its full arguments.

The [connected 128³ MLIR and Spike archive](evidence/nicolas_connected_plain_chain_128_266c593/index.json)
binds both captured sites to Nicolas's A1/B1/B2 codes and E8M0 scales. Its
typed `mx_gemmini.contract` produces quantized C1 codes and scales, which the
SSA-bound `mx_gemmini.resident_contract` consumes. The physical schedule
loads A1/B1, requantizes C1 into the tiled scratchpad and on-chip activation
scale window, then loads only B2 for MM2. A diagnostic readback checks C1
without making MM2 reload it. The generated RV64 ELF matches **16,384 C1
codes, 512 C1 scales, 16,384 C2 codes, and 512 C2 scales** against Nicolas's
checked-in header on stock Spike. An independent build reproduces the MLIR,
commands, objects, ELF, extension, and log hashes. This is source-bound to
the 128³ E4M3 plain MX recipe; arbitrary connected graphs still need a
graph-wide lifetime planner.

The MM1 emitter now consumes a target-derived resident-pair plan: M/N/K
determine A, B, C1, and C2 row spans, transfer counts, scale bytes, and loop
dimensions; the profile determines scratchpad, scale, and accumulator
capacity. The planner rejects overlapping live ranges, misaligned rows, and
insufficient memory. Its source weight layout emitter remains scoped to
square N/K weights. The plan
emits the same 128³ command bytes as the archived Spike run.
The [published planner rerun](evidence/nicolas_resident_pair_plan_b1b5882/index.json)
checks this from commit `b1b5882` on Nicolas's stock Spike: the generated
MLIR, physical program, objects, ELF, and log hashes remain equal to the
connected-chain baseline, with every C1 and C2 output matched.

The [64×128×128 connected chain archive](evidence/nicolas_connected_plain_chain_64x128_d512fc2/index.json)
qualifies a second output shape with the same plain MX profile and Nicolas's
stock Spike extension. Current model2MLIR `e9ded36` captures two PyTorch
matmul sites with `A` shaped 64×128 and both weight matrices shaped 128×128.
The compiler binds them to the first 64 independent rows of Nicolas's
checked-in 128³ packed inputs and C1/C2 goldens. A1's four E8M0 scale groups
are cropped per group; B1 and B2 stay complete. The derived plan emits 32
A1 tile transfers, 64 each for B1/B2, and 32 tile readouts for each result.
MM2 consumes C1 directly from scratchpad. Two captures and two independent
RV64 builds reproduce identical manifests, objects, ELF, and Spike log. Both
runs match **8,192 FP8 codes and 256 E8M0 scales at each site**. The archive
includes the typed MLIR, physical C issuer, source bytes, ELF, hashes, and
full-output Spike result. This qualifies a source row-prefix specialization;
it does not assert that Nicolas checked in a separate 64×128×128 C driver.
A [fresh checkout of the published compiler](evidence/nicolas_connected_plain_chain_64x128_fresh_5cf6e1a/index.json)
replayed the archived frontend capture and reproduced the generated program,
ELF, and complete Spike result from commit `5cf6e1a`.

The [complete 16-row prefix ladder](evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/index.json)
extends that same source specialization to M = **16, 32, 48, 64, 80, 96,
112, and 128**, with N = K = 128. Six new shapes use a fresh two-site
PyTorch capture from the current model2MLIR HEAD `e9ded36`; the 64 and 128
cases retain their separate archives above. For each new shape, two captures
and two generated RV64 builds match byte for byte. Nicolas's stock Spike
matches every C1 and C2 code and scale: per site, **2,048 through 16,384
FP8 codes** and **64 through 512 E8M0 scales** according to M. Each archived
case contains the frontend capture, typed connected MLIR, packed source
slice, physical issuer, ELF, full Spike log, and digest manifests. The
qualification is for independent row prefixes of one checked-in 128³
source header on one plain MX profile; it does not establish arbitrary N/K,
graph topology, precision, or hardware parity.

The connected FP8 pair now lowers through
[`resident_pair_graph.py`](../mx_gemmini_support/resident_pair_graph.py).
It reads the `contract → readout_quantized → resident_contract` SSA graph,
checks all six tensor arguments and runtime payload hashes, verifies the
profile-backed scratchpad plan, and issues both matrix streams plus readout.
Runtime symbols and contraction site IDs are supplied by the caller rather
than fixed to Nicolas's names. The Nicolas adapter retains the source/header
digest, selected profile, and placement checks. Archived commands for all
eight row counts remain byte-identical; a renamed 96×128×128 graph also
passes the native `mx-gemmini-opt` verifier and the reusable lowerer. This
extracts a shared physical lowering boundary; independent numerical
qualification remains necessary for new graph shapes and profiles.
The [published lowerer Spike replay](evidence/nicolas_generic_pair_1f8c6ab/index.json)
records baseline-identical programs and full C1/C2 output for M = 16, 96,
and 128. A separate checkout of commit `1f8c6ab` replayed M = 96 against
the archived baseline with identical command source, objects, ELF, and log.

The reusable pair also has a direct, data-free RV64 object path. The single
`tools.emit_resident_pair_object` command accepts bound MLIR (plain or
gzip-compressed), a selected RTL-derived profile, the six checked runtime
operand files, and an explicit [buffer ABI](../examples/resident-pair-abi.json).
It validates the native dialect IR, typed SSA graph, source payload digest,
buffer sizes, and scratchpad plan; emits the same physical RoCC issuer; and
builds `mx_issue.o` with **zero embedded operand or golden bytes**. The object
manifest records the ten pointer positions, minimum sizes, layouts, command
hashes, tool hashes, and selected profile. The [object and Spike archive](evidence/nicolas_resident_pair_object_1eebfc5/index.json)
qualifies M = 16, 96, and 128. Each object is the sole MX command issuer in
the linked source comparison harness; Nicolas's stock Spike matches every
C1/C2 FP8 code and E8M0 scale. Two independent M = 96 builds reproduce the
object, linked ELF, and Spike log byte for byte.
A [fresh checkout of published commit `5172afa`](evidence/nicolas_resident_pair_object_fresh_5172afa/index.json)
also rebuilt the 96-row object and replayed it on Spike. The object, linked
ELF, and log hashes match the archived baseline; only commit-bearing receipt
fields changed.

```sh
python -m tools.emit_resident_pair_object \
  --mlir docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96/connected_chain.mlir.gz \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --resources-dir docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96 \
  --abi-json examples/resident-pair-abi.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-pair-object-m96
python -m tools.qualify_resident_pair_object \
  --object-dir /new/mx-pair-object-m96 \
  --frontend-dir docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --source-rows 96 \
  --out-dir /new/mx-pair-object-spike-m96
```

This object path uses source bytes to specialize and verify the compiler
input but does not link those bytes into the object. The Spike harness is a
separate source parity gate. This plain FP8 path qualifies the direct
requantized pair on the selected profile and row counts.

The [16×96×96 connected pair archive](evidence/nicolas_plain_pair_16x96_derived_f460d91/index.json)
adds a changed N/K width to the plain MX path. A fresh checkout of
`f460d91` captured both contractions through model2MLIR `e9ded36`, compiled
the typed `contract → readout_quantized → resident_contract` graph, emitted a
data-free RV64 object, and linked that object as the sole MX issuer. Nicolas's
stock Spike matched **1,536 C1 and 1,536 C2 FP8 codes**, plus **48 scales per
site**, with zero mismatches. The inputs are a 96-wide slice of the checked-in
128³ source wire data. The pinned Nicolas mesh model first reproduced the
unchanged 128³ C1 and C2 source goldens; it then produced the 96-wide reference.
This is a source-derived numerical check for one new shape, not an unchanged
96-wide source golden or RTL/FPGA qualification. The archived capture, input
bytes, generated issuer, object, ELF, and Spike log carry SHA-256 digests.
The [64×96×96 fresh-checkout archive](evidence/nicolas_plain_pair_64x96_derived_4dd6108/index.json)
extends the same source-derived path across four M tiles. A checkout of
`4dd6108` reproduced the two-site capture, issuer, data-free object, linked
ELF, and Spike log byte for byte. The pinned model again matched the
unchanged 128-wide source goldens before deriving the 96-wide reference;
Spike matched **6,144 C1 and 6,144 C2 codes** and **192 scales per site**.
The original 16-row fixture remains unchanged. These two cases qualify the
plain MX RoCC/Spike path for their selected shapes and profile.

```sh
python -m tools.capture_nicolas_chain \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQ_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt \
  --matrix-dim 96 --output-rows 16 --out-dir /new/mx-front-16x96
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --connected-frontend-dir /new/mx-front-16x96 \
  --source-rows 16 --source-width 96 --out-dir /new/mx-reference-16x96
python -m tools.compile_object \
  --mlir /new/mx-reference-16x96/connected_chain.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --resources-dir /new/mx-reference-16x96/build \
  --abi-json examples/resident-pair-abi.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-object-16x96
python -m tools.qualify_resident_pair_object \
  --object-dir /new/mx-object-16x96 --frontend-dir /new/mx-front-16x96 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --source-rows 16 --source-width 96 --out-dir /new/mx-object-spike-16x96
```

For the 64-row case, use `--output-rows 64` during capture and
`--source-rows 64` in both Spike qualifiers, with distinct output directories.

The [rectangular connected-pair archive](evidence/nicolas_rectangular_pair_64x96x64_64x64x96_c0be9b9/index.json)
adds **MM1 64×96×64 → MM2 64×64×96**. Nicolas's checked-in
`matmul_fp8_64x96x64` A/B wire inputs and unchanged C1 codes, scales, and
BF16 golden qualify the first stage. B2 is a 96×64 slice of the checked-in
128³ chain weight; the pinned Nicolas mesh model derives C2. A fresh checkout
of `c0be9b9` captured both sites through model2MLIR `e9ded36`, bound their
distinct tensor shapes, emitted a data-free RV64 object, and linked it as the
only MX issuer. Stock Spike matched **6,144 C1 codes and 192 scales**, then
**4,096 C2 codes and 128 scales**, with zero mismatches. The first attempt
had 3,992 C2 code and 34 scale mismatches because MM2 reused the square
chain's B traversal. The fixed rectangular path uses Nicolas's K-major
weight tile order; its regression test and both Spike logs are archived.
This qualifies the selected Rocket RoCC/Spike configuration, not RTL or FPGA.
The archived `bound/connected.mlir.gz` and six `bound/*.bin.gz` files can be
passed directly to `tools.compile_object`; a replay reproduced the archived
object, C issuer, and physical command stream byte for byte.

The [narrow rectangular archive](evidence/nicolas_rectangular_pair_64x96x64_64x32x96_ea746a2/index.json)
repeats capture → bind → object → stock Spike from a fresh `ea746a2`
checkout with `--second-width 32` and separate output directories. It keeps
Nicolas's unchanged 64×96×64 MM1 golden and slices B2 to 96×32. The B1 and
B2 buffers are **6,144 and 3,072 bytes**, with distinct scratchpad tail
placements. Spike matched **6,144 C1 codes, 192 C1 scales, 2,048 C2 codes,
and 64 C2 scales** against the checked source/model references. The first
build and fresh replay produced identical captured MLIR, bound payloads,
object, ELF, and Spike log. The archive hashes every replay artifact and
includes the data-free object and complete output reference.

```sh
python -m tools.capture_nicolas_chain \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQ_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt \
  --matrix-dim 96 --first-k 64 --second-width 64 --output-rows 64 \
  --out-dir /new/mx-rectangular-capture
python -m tools.bind_rectangular_chain \
  --capture-dir /new/mx-rectangular-capture --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-rectangular-bound
python -m tools.compile_object \
  --mlir /new/mx-rectangular-bound/connected.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --resources-dir /new/mx-rectangular-bound \
  --abi-json examples/resident-pair-abi.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-rectangular-object
python -m tools.qualify_rectangular_pair_object \
  --object-dir /new/mx-rectangular-object \
  --capture-dir /new/mx-rectangular-capture \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-rectangular-spike
```

The MX+VPU 64³ connected graph now has a parallel object path through
[`resident_vpu_graph.py`](../mx_gemmini_support/resident_vpu_graph.py). It
checks the `contract → readout_bf16 → vpu_execute → spad_requant →
resident_contract` SSA edges, six input lengths and payload hashes, VPU
scratchpad lifetime, MM2 placement, and the selected MX+VPU profile. The
caller supplies runtime symbols and site IDs. The source adapter continues
to check Nicolas's header, first-matmul capture, VPU seam, and output goldens.
The command stream and `mx_issue.o` are byte-identical to the established
source path. The [Spike and fresh-checkout archive](evidence/nicolas_resident_vpu_object_6c9ed40/index.json)
records 0 mismatches across 4,096 BF16 values, 8,192 FP8 codes, and 256
E8M0 scales on Nicolas's stock Spike for
`MxE4M3Fp4VpuGemminiRocketConfig`. The object carries no operand or golden
data. This numerical qualification covers the 64³ scalar ×2 chain; other
VPU operations, shapes, and precisions need their own tests.

The [narrow MX+VPU archive](evidence/nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c/index.json)
adds a captured 64×64×64 MM1 followed by VPU ×2, resident requant, and a
64×32×64 MM2. It uses Nicolas's unchanged 64³ chain source and its first
32 B2 columns. The first output block is checked twice: against the source
codes/scales and against independently requantized source BF16. From a fresh
`9cd918c` checkout, the typed graph compiled to a data-free RoCC object and
stock Spike matched **4,096 C1 BF16 values, 4,096 C1 codes, 128 C1 scales,
2,048 C2 codes, and 64 C2 scales**. The captured MLIR, bound payloads,
physical program, object, ELF, and Spike log matched the first run byte for
byte. The archived compressed MLIR and inputs also recompile to the same
object, issuer, and physical program.

The same narrow graph was independently captured, bound, compiled, and run
under Nicolas's **E4M3-only VPU** profile. Its
[configuration receipt](evidence/nicolas_narrow_vpu_pair_e4m3_only_103acdc/index.json)
records the distinct profile hash and zero mismatches on stock Spike for all
the outputs above. The source payloads, generated issuer/object, ELF, and
Spike log match the FP4-capable VPU profile's FP8 run byte for byte; the
physical program receipt differs in its profile hash. This qualifies the
shared FP8/VPU path on both selected Rocket profiles. FP4 execution remains
legal only on the FP4-capable profile.

To reproduce, capture with `--matrix-dim 64 --first-k 64 --second-width 32
--output-rows 64` and the VPU profile, then use `tools.bind_narrow_vpu_chain`,
`tools.compile_object` with `examples/resident-vpu-abi.json`, and
`tools.qualify_narrow_vpu_object`. Each command requires a new output
directory. The bind and Spike qualifiers check the source, profile,
model2MLIR receipt, runtime payloads, and object hashes.

To reproduce from a checked-out compiler and Nicolas's source tree, first
run `tools.qualify_nicolas_vector_requant` with `--connected-ssa` and the
archived two-site frontend receipt to materialize checked input `.bin` files.
Then use the generated `connected_bound.mlir` and `build/` directory:

```sh
python tools/qualify_nicolas_vector_requant.py \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --with-resident-matmul --connected-ssa \
  --frontend-bound-mlir docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/frontend_bound.mlir \
  --frontend-receipt docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/capture_receipt.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-vpu-source
python -m tools.emit_resident_vpu_object \
  --mlir /new/mx-vpu-source/connected_bound.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --resources-dir /new/mx-vpu-source/build \
  --abi-json examples/resident-vpu-abi.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-vpu-object
python tools/qualify_nicolas_vector_requant.py \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --with-resident-matmul --connected-ssa \
  --frontend-bound-mlir docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/frontend_bound.mlir \
  --frontend-receipt docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/capture_receipt.json \
  --issuer-object /new/mx-vpu-object/mx_issue.o \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-vpu-object-spike
```

### One object compiler entry point

[`tools.compile_object`](../tools/compile_object.py) selects the object
lowerer from the verified typed graph. A source-bound single contraction uses
`--bundle`; a connected resident pair uses `--resources-dir` and
`--abi-json`. The driver accepts plain or gzip-compressed MLIR for resident
graphs and writes `compile_manifest.json` beside the existing object and
physical program manifests. It rejects ambiguous bindings and graph families
without executable lowering. This dispatch does not change scheduling:
`emit_mx_object`, `emit_resident_pair_object`, and
`emit_resident_vpu_object` still own their respective physical streams.

For example, a source-bound FP4 contraction can be compiled in one command:

```sh
python -m tools.compile_object \
  --mlir docs/evidence/radiance_plain_mx_profile_trio_266c593/fp4/payload_bound.mlir \
  --bundle docs/evidence/radiance_plain_mx_profile_trio_266c593/fp4/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-fp4-object
```

For a resident graph, replace `--bundle` with `--resources-dir` and the
matching `examples/resident-pair-abi.json` or
`examples/resident-vpu-abi.json`. The source bundle or input files are needed
to verify the typed payload; the emitted object contains no operand or golden
data. Existing Spike qualifiers link that object and compare full outputs.
`tools.compile_mx --issuer-object /new/mx-fp4-object/mx_issue.o` links a
source-bound object into the standalone source-golden harness after checking
its MLIR, bundle, profile, generated C, and object hashes. The same route
works for FP4, FP6, and FP8 on `MxGemminiRocketConfig`; their full-output
Spike logs match the established source builds. The
[fresh compiler and source-parity archive](evidence/mx_compile_object_dispatch_d3156e4/index.json)
records the object hashes, physical programs, and numerical checks.

For example, replay the archived 96×128×128 capture without recapturing
PyTorch:

```sh
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --connected-frontend-dir docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96 \
  --source-rows 96 --out-dir /new/mx-connected-96x128 \
  --baseline-manifest docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96/artifact_manifest.json
```

```sh
python -m tools.capture_nicolas_chain \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQ_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt \
  --matrix-dim 128 --output-rows 64 \
  --out-dir /new/mx-front-64x128
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --connected-frontend-dir /new/mx-front-64x128 --source-rows 64 \
  --out-dir /new/mx-connected-64x128 \
  --baseline-manifest docs/evidence/nicolas_connected_plain_chain_64x128_d512fc2/artifact_manifest.json
```

```sh
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --connected-frontend-dir docs/evidence/nicolas_plain_chain_128_model2mlir_e9ded36 \
  --out-dir /new/mx-connected-128 \
  --baseline-manifest docs/evidence/nicolas_connected_plain_chain_128_266c593/artifact_manifest.json
```

A [fresh GitHub checkout of `ae945d0`](evidence/nicolas_connected_plain_chain_128_fresh_checkout_ae945d0/index.json)
built `mx-gemmini-opt` from source and ran that exact command with Nicolas's
RTL and RISC-V toolchain. It reproduced the connected MLIR, physical program,
objects, ELF, and Spike log hashes from the checked-in baseline. The archive
records the fresh compiler revision, native tool hash, artifact manifest, and
full-output Spike result.

```sh
git clone --single-branch --branch handwritten-implementation \
  git@github.com:ucb-bar/mx-gemmini-mlir.git /new/mx-gemmini-mlir
git -C /new/mx-gemmini-mlir checkout --detach \
  ae945d0d10e1234945b2f7bb68242fed54a9295b
cmake -S /new/mx-gemmini-mlir -B /new/mx-gemmini-mlir/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR="$MLIR_DIR"
cmake --build /new/mx-gemmini-mlir/build --target mx-gemmini-opt
cd /new/mx-gemmini-mlir
python -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --connected-frontend-dir docs/evidence/nicolas_plain_chain_128_model2mlir_e9ded36 \
  --out-dir /new/mx-connected-128-run \
  --baseline-manifest docs/evidence/nicolas_connected_plain_chain_128_266c593/artifact_manifest.json
```

A [current upstream frontend rerun](evidence/nicolas_connected_chain_upstream_e9ded36_20261010/README.md)
recaptured the two-site graph with model2MLIR `e9ded36` and executed the
connected SSA chain on the same pinned Spike model. The new frontend and
connected MLIR hashes differ from the earlier `7485a829` capture, while the
issued objects, ELF, extension, and numerical log remain byte-identical.
The [receipt test](../tests/test_current_nicolas_connected_chain.py) checks
both quantized sites, the connected artifacts, and all output counts.

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
Add `--connected-ssa` to emit and compile the single SSA-connected function.
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

### E4M3 activation with E2M3 weights

The standalone `MxAsymE4M3E2M3GemminiRocketConfig` selects PE mode 9.
Both operands use 8-bit LUT entries; the weight codes use the low six bits.
The compiler emits the per-operand alternate-format bit in `CONFIG_EX` and
loads Nicolas's checked-in E4M3 and E2M3 LUT banks. The generated physical
program has 63 commands. Its standalone ELF matches **all 4,096 BF16 source
values** on the pinned Spike extension, and two fresh captures and builds
match on frontend artifacts, resources, commands, issuer, objects, ELF,
extension, and Spike log hashes: [receipt](evidence/compiled_nicolas_asym_physical_e4m3_e2m3_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_physical_e4m3_e2m3_repro_20261009.json),
[bound MLIR](evidence/model2mlir_nicolas_asym_e4m3_e2m3_bound_20261009.mlir),
[source recipe](evidence/model2mlir_nicolas_asym_e4m3_e2m3_recipe_20261009.json),
and [physical program](evidence/compiled_nicolas_asym_physical_e4m3_e2m3_program_20261009.json).
As with the other mixed modes, model2MLIR captures the PyTorch matmul site;
the explicit checked-in source recipe supplies its mixed precision and packed
data. This standalone profile has no VPU.

```sh
python -m tools.qualify_nicolas_asym --variant e4m3_e2m3_lut \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAsymE4M3E2M3GemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/output-directory
```

### E5M2 activation with FP4 weights

Nicolas's `MxAsymE5M2Fp4GemminiRocketConfig` selects E5M2 activation through
an 8-bit LUT, direct FP4 weights, and PE mode 3. This exercises the FP8
alternate-format bit in `CONFIG_EX` independently of the E4M3 cases. The
source-bound typed contraction lowers to 63 physical commands. Two fresh
model2MLIR captures and builds each match **all 4,096 BF16 source values**
on the pinned Spike extension; their bound MLIR, resource manifests,
command streams, objects, ELF, and Spike logs have matching hashes:
[receipt](evidence/compiled_nicolas_asym_payload_e5m2_fp4_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_payload_e5m2_fp4_repro_20261009.json),
[typed MLIR](evidence/model2mlir_nicolas_asym_payload_e5m2_fp4_bound_20261009.mlir),
and [physical program](evidence/compiled_nicolas_asym_payload_e5m2_fp4_program_20261009.json).
The source recipe supplies packed data and mixed precision; model2MLIR
captures the PyTorch matmul graph. This standalone profile has no VPU.

```sh
python -m tools.qualify_nicolas_asym --variant e5m2_fp4_lut \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAsymE5M2Fp4GemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/output-directory
```

### Direct FP8 operand layouts

Two more Nicolas modes require different transfer geometry. E4M3 direct
activation × E3M2 LUT weight uses a full 64×64 activation array and only a
weight LUT. FP4 direct activation × E4M3 direct weight uses a full 64×64
weight array and four N tiles instead of two. The physical lowerer now derives
the activation and weight transfer counts and strides from the selected
source layout. Each generated 69-command program matched **all 4,096 BF16
source values** in two independent post-commit Spike runs.

| Standalone profile and source mode | PE mode | Evidence |
|---|---:|---|
| `MxAsymE4M3E3M2GemminiRocketConfig`, `--variant e4m3_direct_e3m2` | 7 | [receipt](evidence/compiled_nicolas_asym_payload_e4m3s_e3m2_20261009.json), [reproduction](evidence/compiled_nicolas_asym_payload_e4m3s_e3m2_repro_20261009.json), [program](evidence/compiled_nicolas_asym_payload_e4m3s_e3m2_program_20261009.json) |
| `MxAsymFp4E4M3GemminiRocketConfig`, `--variant fp4_direct_e4m3` | 2 | [receipt](evidence/compiled_nicolas_asym_payload_fp4_e4m3s_20261009.json), [reproduction](evidence/compiled_nicolas_asym_payload_fp4_e4m3s_repro_20261009.json), [program](evidence/compiled_nicolas_asym_payload_fp4_e4m3s_program_20261009.json) |

Both use the same CLI arguments shown above with their respective profile
path. Their typed MLIR binds the source [E4M3×E3M2](evidence/model2mlir_nicolas_asym_payload_e4m3s_e3m2_bound_20261009.mlir)
or [FP4×E4M3](evidence/model2mlir_nicolas_asym_payload_fp4_e4m3s_bound_20261009.mlir)
resource manifest before physical lowering. These are standalone MX profiles
without VPU.

### Typed binding of Nicolas's packed source resources

The asymmetric CLI now materializes an external resource manifest before
physical lowering. It records each checked-in operand, E8M0 scale, LUT,
and BF16 golden array with its byte hash, shape, element width, and layout.
The manifest digest is attached to the typed `mx_gemmini.contract` and its
module, with an explicit Nicolas source origin. Both the Python verifier and
`mx-gemmini-opt` check this binding. The physical lowerer regenerates the
manifest from the named source and refuses a missing or changed binding.
The byte arrays remain separate compiler input files; the PyTorch capture
still supplies the contraction graph and does not claim to have quantized
those arrays.

The E4M3×E2M3 mode passed **4,096 / 4,096 BF16** values twice with this
binding; the two runs match on the bound MLIR, manifest, command stream,
objects, ELF, and Spike log hashes. See the [receipt](evidence/compiled_nicolas_asym_payload_e4m3_e2m3_20261009.json),
[reproduction](evidence/compiled_nicolas_asym_payload_e4m3_e2m3_repro_20261009.json),
[typed MLIR](evidence/model2mlir_nicolas_asym_payload_e4m3_e2m3_bound_20261009.mlir),
and [resource manifest](evidence/compiled_nicolas_asym_payload_e4m3_e2m3_resources_20261009.json).
The reverse FP4×FP6 mode also passed a fresh [Spike regression](evidence/compiled_nicolas_asym_payload_fp4_fp6_regression_20261009.json)
with the binding.

### Dedicated DIM16 asymmetric mode matrix

The matrix CLI discovers Nicolas's 64×64×64 asymmetric tests, derives each
mode from its named source/header pair, and checks it against the matching
RTL-derived profile. Preflight requires every legal compute cell in the 20
dedicated DIM16 `MxAsym*GemminiRocketConfig` profiles to have exactly one
source variant. Removing a source test makes preflight fail. The CLI captures
a fresh PyTorch matmul with the pinned model2MLIR, binds the checked-in
operand, scale, LUT, and BF16 golden bytes to typed MLIR, emits a physical
command program and standalone RV64 ELF, then compares all source outputs
on Nicolas's pinned Spike extension.

Two independent post-commit matrix runs passed **26 / 26 legal modes** and
**106,496 / 106,496 BF16 values per run**. For every mode, the captured
frontend artifacts, source resource manifest, physical program, generated
issuer and driver, RV64 objects and ELF, Spike extension, and Spike log
hashes agree across runs. Only link log hashes differ because their paths
differ. The [first matrix receipt](evidence/nicolas_asym_matrix_dim16_266c593/matrix_first.json)
and [reproduction](evidence/nicolas_asym_matrix_dim16_266c593/matrix_repro.json)
index all 52 [per-mode receipts](evidence/nicolas_asym_matrix_dim16_266c593/)
and record the selected PE tuple for each mode. They pin compiler revision
`338faa8`, Nicolas RTL `266c593`, and the source/tool hashes.

```sh
python -m tools.qualify_nicolas_asym_matrix \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/matrix-output --jobs 4
```

This matrix qualifies the dedicated DIM16 Rocket profiles and their BF16
64³ source tests. The following sections separately qualify the available
all-asymmetric source modes. MX+VPU evidence appears above.

### DIM8 and DIM32 all-asymmetric source matrices

The same compiler path now derives transfer geometry, scratchpad row ranges,
and loop tile counts from `MxDim8AllAsymGemminiRocketConfig` and
`MxDim32AllAsymGemminiRocketConfig`. It binds each named Nicolas source/header
pair to typed MLIR, emits an RV64 RoCC ELF, and compares every BF16 output
against the checked-in source golden using the matching `gemmini_dim8` or
`gemmini_dim32` Spike extension.

Two fresh runs for **each mesh** passed **21 / 21 available source modes** and
**86,016 / 86,016 BF16 values per run**. The frontend artifacts, bound payloads,
physical programs, generated C, objects, ELFs, extensions, and Spike logs
match across each pair of runs. Only path-bearing link log hashes differ.
The [DIM8 first run](evidence/nicolas_asym_matrix_dim8_266c593/matrix_first.json),
[DIM8 reproduction](evidence/nicolas_asym_matrix_dim8_266c593/matrix_repro.json),
[DIM32 first run](evidence/nicolas_asym_matrix_dim32_266c593/matrix_first.json),
and [DIM32 reproduction](evidence/nicolas_asym_matrix_dim32_266c593/matrix_repro.json)
index 84 per-mode receipts and pin compiler `de188c1`, Nicolas RTL `266c593`,
model2MLIR `7485a82`, and MXQuant `b4af543`.

```sh
python -m tools.qualify_nicolas_asym_matrix --mesh-dim 8 \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/dim8-matrix-output --jobs 4
# Repeat with --mesh-dim 32 and a new output directory.
```

Each all-asymmetric profile has **36 legal compute cells**. Nicolas checks in
64³ source tests for 21 of them. The
[generation wrapper](../tools/generate_nicolas_mesh_headers.py) regenerates a
checked-in header byte for byte at each mesh before obtaining the 15 missing
mode tests from his pinned `gen_asym.py` and microxcaling `7bc41952`. On
DIM32, five of those headers already existed without named source C tests;
the DIM8 headers are all newly generated. The
[DIM8 header manifest](evidence/nicolas_generated_mesh_dim8_266c593/generation_manifest.json)
and [DIM32 header manifest](evidence/nicolas_generated_mesh_dim32_266c593/generation_manifest.json)
record their origins and hashes. Each mode is bound to a fresh model2MLIR
matmul capture; the compiler emits the physical MX commands and standalone
RV64 RoCC ELF. No handwritten replacement contraction enters the output.

The full 36-mode stock Spike matrix passed **35 / 36** modes on both DIM8 and
DIM32. Every passing test compared **4,096 / 4,096 BF16 outputs** against its
source header, for **143,360 matching outputs per mesh**. An independent
15-mode generated-mode matrix reproduced the 14 passing modes on each mesh;
the frontend, bound payload, physical program, issuer, ELF, and Spike log
hashes agree between runs after omitting the path-bearing link log hash. The
[DIM8 full matrix](evidence/nicolas_generated_mesh_dim8_266c593/matrix_first.json),
[DIM8 generated reproduction](evidence/nicolas_generated_mesh_dim8_266c593/matrix_generated_repro.json),
[DIM32 full matrix](evidence/nicolas_generated_mesh_dim32_266c593/matrix_first.json),
and [DIM32 generated reproduction](evidence/nicolas_generated_mesh_dim32_266c593/matrix_generated_repro.json)
pin compiler `d8d2f55`, Nicolas RTL `266c593`, model2MLIR `7485a82`,
MXQuant `b4af543`, source headers, and simulator hashes. The
[DIM8 qualification index](evidence/nicolas_generated_mesh_dim8_266c593/qualification.json)
and [DIM32 qualification index](evidence/nicolas_generated_mesh_dim32_266c593/qualification.json)
link all 36 first-run receipts and 15 generated-mode reruns per mesh.

The sole failing mode on both meshes is **direct E4M3 activation × E4M3 LUT
weights**. The stock model reports **4,095 mismatches on DIM8** and **4,096
on DIM32**. The same respective compiler ELFs report **zero mismatches** with
the [isolated Spike model correction](evidence/nicolas_generated_modes_266c593/spike_weight_lut_quad_candidate.patch)
that includes RTL `weight_lut_en` in weight lane selection. The
[DIM8 diagnostic](evidence/nicolas_generated_mesh_dim8_266c593/e4m3s_e4m3/patch_diagnostic.json)
and [DIM32 diagnostic](evidence/nicolas_generated_mesh_dim32_266c593/e4m3s_e4m3/patch_diagnostic.json)
include both extension and log hashes. That change is not merged into
Nicolas's Spike branch. These two legal cells remain unqualified on the
pinned stock model. These 64³ BF16 tests do not establish general PyTorch
operand quantization, arbitrary shapes, quantized readouts, timing, or FPGA.

```sh
python -m tools.generate_nicolas_mesh_headers --mesh-dim 8 \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --microxcaling-root /path/to/microxcaling-at-7bc41952
python -m tools.qualify_nicolas_asym_matrix --mesh-dim 8 \
  --include-generated --source-shape 64x64 \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/dim8-generated-matrix-output --jobs 4
# Repeat both commands with --mesh-dim 32 and a new output directory.
```

### DIM16 all-asymmetric source matrix

Nicolas's `MxAllAsymGemminiRocketConfig` selects one DIM16 Rocket profile with
36 legal compute cells. All 26 checked-in DIM16 asymmetric 64³ source/header
pairs bind to this profile. The matrix command selects it with `--all-asym`;
without that flag, DIM16 selects the dedicated per-pair profiles above.

Two independent runs passed **26 / 26 available source modes** and
**106,496 / 106,496 BF16 outputs per run** on the pinned DIM16 Spike
extension. Every frontend artifact, bound payload, physical program, generated
source, object, ELF, extension, and Spike log hash agrees across runs. Only
path-bearing link log hashes differ. The [first matrix receipt](evidence/nicolas_asym_matrix_dim16_all_266c593/matrix_first.json)
and [reproduction](evidence/nicolas_asym_matrix_dim16_all_266c593/matrix_repro.json)
index 52 per-mode receipts and pin compiler `0e03168` and Nicolas RTL
`266c593`.

```sh
python -m tools.qualify_nicolas_asym_matrix --mesh-dim 16 --all-asym \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/dim16-all-asym-output --jobs 4
```

These asymmetric-only aggregate receipts enumerate the **10 legal cells**
outside their source set. They qualify the 64³ BF16 cases they name on Spike;
arbitrary shapes remain unqualified.

Nicolas also checks in same-format E2M3, E4M3, and E5M2 LUT source tests with
BF16 goldens. `--include-symmetric-lut` adds these three tests to the DIM16
all-asymmetric matrix. Two fresh runs passed **29 / 29 source modes** and
**118,784 / 118,784 BF16 outputs per run**. The per-mode frontend, payload,
physical program, generated source, ELF, extension, and Spike log hashes
reproduce; only path-bearing link log hashes differ. The
[combined first receipt](evidence/nicolas_asym_matrix_dim16_all_plus_symmetric_266c593/matrix_first.json)
and [reproduction](evidence/nicolas_asym_matrix_dim16_all_plus_symmetric_266c593/matrix_repro.json)
index 58 per-mode receipts pinned to compiler `4c4fa5e` and RTL `266c593`.
The aggregate lists the **seven legal cells still without BF16 source tests**.

```sh
python -m tools.qualify_nicolas_asym_matrix --mesh-dim 16 --all-asym \
  --include-symmetric-lut --source-shape 64x64 \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/dim16-combined-output --jobs 4
```

Nicolas also checks in a direct FP4×FP4 mode-0 BF16 source test. Adding
`--include-symmetric-fp4` to the command above passes **30 / 30 selected
modes** and **122,880 / 122,880 BF16 outputs** on the pinned DIM16 Spike.
The [30-mode matrix receipt](evidence/nicolas_asym_matrix_dim16_all_plus_fp4_266c593/matrix_first.json)
indexes its per-mode receipts at compiler `b6019d0` and RTL `266c593`.
An [independent FP4-only run](evidence/nicolas_asym_matrix_dim16_all_plus_fp4_266c593/matrix_fp4_repro.json)
reproduces the new mode's frontend, physical program, ELF, and Spike log;
only its path-bearing link log hash differs. The older 29-mode subset has the
two full reproductions above. The 30-mode aggregate enumerates **six legal
cells without checked-in Nicolas BF16 source tests**. It does not imply
those cells are unsupported by RTL.

### Radiance direct FP8 on the all-asymmetric profile

The Radiance `mxgemm.fp8.singletile.tm64tn64tk64.fullout.cpp` driver at
`d4732fb` provides a separate direct E4M3×E4M3 (`pe_mode=8`) test. Its
64×64×64 data header is generated by the pinned Radiance
`gen_mxgemm_data.py` and `lib/golden/mx_golden`; the generated header is
[archived](evidence/radiance_fp8_dim16_all_asym_266c593/generated_source_header.h)
with its hash. Latest model2MLIR `7485a829` captures the PyTorch matmul;
the MX binder attaches the generated source codes and E8M0 scales to the
profile-bound contraction. The compiler emits the
[physical command stream](evidence/radiance_fp8_dim16_all_asym_266c593/physical_program.json)
and Rocket RoCC issuer. Two independent Spike builds each match
**4,096 / 4,096 BF16 outputs** against the generated source golden. Their
[first](evidence/radiance_fp8_dim16_all_asym_266c593/first.json) and
[reproduction](evidence/radiance_fp8_dim16_all_asym_266c593/repro.json)
manifests match after omitting only the path-bearing link log hash. The
[frontend receipt](evidence/radiance_fp8_dim16_all_asym_266c593/capture_receipt.json)
pins the Radiance, model2MLIR, and MXQuant revisions. This qualifies one
additional legal DIM16 compute cell beyond the 30 checked-in Nicolas cases;
five cells now lack full-output source evidence across these two source sets.
It is a Rocket/Spike result, not a Radiance FPGA result.

```sh
# In an isolated Radiance worktree at d4732fb:
make -C lib/golden mx_golden
python kernels/gemm_mxgemmini/gen_mxgemm_data.py fp8 64 64 64
# In mx-gemmini-mlir, capture with tests/capture_radiance_mx_gemm.py using
# MxAllAsymGemminiRocketConfig.json, then use tools.qualify_source_mx on
# mx_gemm.profile_bound.mlir and the Radiance fullout driver.
```

### Generated DIM16 mode probes

Five other legal cells lack a checked-in Nicolas fullout driver. The
[generation wrapper](../tools/generate_nicolas_missing_headers.py) registers
their format pairs with Nicolas's pinned `gen_asym.py` without changing its
quantization or matrix model. Before emitting new headers, it regenerates a
checked-in E2M3×E4M3 header byte for byte. Its
[manifest](evidence/nicolas_generated_modes_266c593/mx_gemmini_generated_modes_manifest.json)
pins the generator, baseline hash, RTL and software revisions, and the
upstream microxcaling revision `7bc41952`. The new headers are archived with
their hashes. `tools.qualify_nicolas_asym --generated-mode <mode>` binds one
such header to a current model2MLIR matmul capture, lowers the physical MX
commands, builds an RV64 Rocket ELF, and runs Nicolas's Spike extension. No
new handwritten C kernel enters the compiled output.

| Activation × weight | Stock Spike BF16 result | Evidence |
|---|---:|---|
| E2M3 LUT × E4M3 direct | 4,096 / 4,096, twice | [first](evidence/nicolas_generated_modes_266c593/e2m3_e4m3s/first.json), [reproduction](evidence/nicolas_generated_modes_266c593/e2m3_e4m3s/repro.json) |
| E3M2 LUT × E3M2 LUT | 4,096 / 4,096, twice | [first](evidence/nicolas_generated_modes_266c593/e3m2_e3m2/first.json), [reproduction](evidence/nicolas_generated_modes_266c593/e3m2_e3m2/repro.json) |
| E4M3 direct × E2M3 LUT | 4,096 / 4,096, twice | [first](evidence/nicolas_generated_modes_266c593/e4m3s_e2m3/first.json), [reproduction](evidence/nicolas_generated_modes_266c593/e4m3s_e2m3/repro.json) |
| E4M3 LUT × E4M3 direct | 4,096 / 4,096, twice | [first](evidence/nicolas_generated_modes_266c593/e4m3_e4m3s/first.json), [reproduction](evidence/nicolas_generated_modes_266c593/e4m3_e4m3s/repro.json) |
| E4M3 direct × E4M3 LUT | **4,094 mismatches** | [failed receipt](evidence/nicolas_generated_modes_266c593/e4m3s_e4m3/first.json), [stock log](evidence/nicolas_generated_modes_266c593/e4m3s_e4m3/stock_spike.log) |

The four passing modes reproduce their frontend and target MLIR, physical
program, C issuer, ELF, and Spike log hashes; only path-bearing link log
hashes differ. With the 30 checked-in Nicolas modes and the separate Radiance
direct FP8 mode, **35 of 36 distinct DIM16 legal cells** now have full-output
parity on unmodified Spike. These counts refer to the 64³ BF16 mode tests;
they do not qualify arbitrary shapes, quantized readouts, timing, or FPGA.
The [qualification index](evidence/nicolas_generated_modes_266c593/qualification.json)
enumerates the exact covered and uncovered compute tuples.

For the failing cell, the pinned Spike extension's direct-E4M3 activation
path chooses single-column weights when format and alternate-format bits are
zero, even if its E4M3 weight LUT is loaded. Nicolas's RTL
`ExecuteController.mx_multi_elem` includes `weight_lut_en` in that lane
decision. Applying the [candidate model patch](evidence/nicolas_generated_modes_266c593/spike_weight_lut_quad_candidate.patch)
in an isolated extension checkout makes the **same compiler ELF** match
4,096 / 4,096 outputs; the [diagnostic](evidence/nicolas_generated_modes_266c593/e4m3s_e4m3/patch_diagnostic.json)
records both extension and log hashes. That patch has not been merged into
Nicolas's Spike branch, and this cell remains unqualified on the pinned model.

```sh
# From mx-gemmini-mlir, with an isolated gemmini checkout at 266c593:
python -m tools.generate_nicolas_missing_headers \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --microxcaling-root /path/to/microxcaling-at-7bc41952
python -m tools.qualify_nicolas_asym --generated-mode e2m3_e4m3s \
  --mesh-dim 16 --model2mlir-root /path/to/model2MLIR \
  --mxq-root /path/to/MXQuant --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxAllAsymGemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/generated-mode-output
```

### Larger asymmetric source shapes

The source-bound scheduler now derives the operand strides, scale byte counts,
tile loops, scratchpad layout, and BF16 readout size from the selected recipe.
Nicolas's direct E4M3 activation × FP4 weight sources at **128×128×128 on
DIM16** and **128×128×256 on DIM32** each compile from a fresh model2MLIR
PyTorch matmul capture to typed MX MLIR, physical commands, and an RV64 RoCC
ELF. Each generated program matches **16,384 / 16,384 BF16 source outputs**
on the corresponding pinned Spike extension. Two independent builds per
shape reproduce all recorded artifacts and Spike logs; only path-bearing
link log hashes differ. The [qualification index](evidence/nicolas_asym_large_direct_266c593/qualification.json)
links the four receipts, captured and bound MLIR, physical programs, resource
manifests, and generated C issuers. It pins compiler `64cb454` and Nicolas
RTL `266c593`.

```sh
python -m tools.qualify_nicolas_asym --source-suffix e4m3s_fp4 \
  --source-shape 128x128x256 --mesh-dim 32 \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim32AllAsymGemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/large-asym-output
```

For DIM16 use `--source-shape 128x128 --mesh-dim 16` and
`MxAllAsymGemminiRocketConfig.json`.

The other three checked-in **128×128×256 DIM32** asymmetric source cases also
compile and match **16,384 / 16,384 BF16 outputs each** on Nicolas's Spike:

| Activation × weight | PE mode | Source LUT banks |
|---|---:|---|
| FP6 E2M3 LUT × FP8 E5M2 LUT | 10 | A, B, C: 64 lines each |
| FP8 E4M3 direct × FP8 E5M2 LUT | 7 | B: 64 lines |
| FP4 direct × FP6 E3M2 LUT | 1 | A, B, C: 64 lines each |

Two independent post-commit runs per case reproduce their captured and bound
MLIR, physical commands, resource manifests, generated C, objects, ELFs,
extensions, and Spike logs. Only path-bearing link log hashes differ. The
[LUT qualification index](evidence/nicolas_asym_large_lut_266c593/qualification.json)
links all six receipts and inspectable generated artifacts; it pins compiler
`563f6eb` and Nicolas RTL `266c593`. Select a row with
`--source-suffix e2m3_e5m2`, `e4m3s_e5m2`, or `fp4_fp6` while retaining
`--source-shape 128x128x256 --mesh-dim 32` and the DIM32 all-asymmetric
profile in the command above. All five checked-in larger asymmetric source
cases now have full-output Spike parity; arbitrary shapes remain unqualified.

The matrix CLI can also run each larger source set with one command:

```sh
python -m tools.qualify_nicolas_asym_matrix \
  --mesh-dim 32 --source-shape 128x128x256 \
  --model2mlir-root /path/to/model2MLIR --mxq-root /path/to/MXQuant \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/dim32-large-matrix --jobs 4
# For the DIM16 case, use --mesh-dim 16 --all-asym --source-shape 128x128.
```

Two independent suite runs passed **4 / 4** DIM32 cases and **1 / 1** DIM16
case, with all 16,384 outputs matched in every case. The
[DIM32 first](evidence/nicolas_asym_large_matrix_266c593/dim32_matrix_first.json)
and [reproduction](evidence/nicolas_asym_large_matrix_266c593/dim32_matrix_repro.json),
plus [DIM16 first](evidence/nicolas_asym_large_matrix_266c593/dim16_matrix_first.json)
and [reproduction](evidence/nicolas_asym_large_matrix_266c593/dim16_matrix_repro.json),
index ten per-mode receipts. They pin compiler `56a9574` and
record the 32 and 35 legal cells, respectively, outside these *larger-shape*
source sets. The 64³ matrices above qualify additional cells in the same
profiles; these missing lists are specific to each source shape.

## Latest complete Radiance MX GEMM roster

### Nicolas's plain MX Rocket profile across FP8, FP4, and FP6

The [plain-profile qualifier](../tools/qualify_radiance_mx_base_profile.py)
rebinds three archived model2MLIR `e9ded36` captures to Nicolas's
`MxGemminiRocketConfig` at RTL `266c593`. This is the MX-only Rocket
configuration with three legal compute modes and no VPU. The selected
Radiance sources are FP8 128×128×512, FP4 64×64×128, and FP6 128×128×2048.
The qualifier checks each driver and data header against the capture roster,
binds actual packed codes, E8M0 scales, and, for FP6, all three 64-line LUT
banks. It lowers the typed contractions to physical commands, builds RV64
ELFs, and compares every BF16 output on Nicolas's pinned Spike extension.

Two independent runs from compiler `48fbb89` matched **16,384 FP8**, **4,096
FP4**, and **16,384 FP6** BF16 outputs, **36,864 total**. A third run used a
fresh Radiance `80f84ca` worktree after generating only missing headers; its
bound MLIR, bundles, physical commands, generated C, ELFs, and Spike logs
match the first two byte for byte. The
[qualification index](evidence/radiance_plain_mx_profile_trio_266c593/index.json),
[fresh materialization receipt](evidence/radiance_plain_mx_profile_trio_266c593/fresh_materialization.json),
and [regression test](../tests/test_plain_mx_profile_trio_evidence.py)
record the inputs and all three executions. Exact ELF bytes are archived for
each precision; only path-bearing link log hashes differ between builds.
This qualifies the three selected mode/shape combinations on the plain
Rocket profile's pinned functional simulator. It does not certify every
profile, RTL timing, or an FPGA image.

From a fresh compatible Radiance checkout, build its golden generator and
materialize missing headers before the MX qualification command:

```sh
make -C "$RADIANCE_KERNELS_ROOT/lib/golden" mx_golden
python -m tools.materialize_radiance_roster_headers \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --expected-index docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend/index.json \
  --compatible-source-revision --out /tmp/mx-header-materialization.json
python -m tools.qualify_radiance_mx_base_profile \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --out-dir /tmp/mx-plain-profile-trio
```

The materializer preserves committed headers and checks all 31 driver/header
hashes. The qualifier itself compiles only the three cases above and refuses
changed source bytes or a drifted RTL profile.

### Selected legal modes in six additional MX Rocket profiles

The same qualifier accepts `--profile` and repeatable `--case` arguments.
It preflights every selected model2MLIR handoff against the target profile
before writing output, so unsupported format/profile combinations fail closed.
On Nicolas's pinned Spike, two independent builds of each selected case from
compiler `82a8bae` produced identical bound MLIR, physical commands, C
issuers, ELFs, and simulator logs. All **147,456 / 147,456 BF16 outputs**
matched the checked Radiance source goldens:

| RTL profile | Source cases | Compared BF16 outputs |
|---|---|---:|
| `MxFp4OnlyGemminiRocketConfig` | FP4 | 4,096 |
| `MxE4M3SingleGemminiRocketConfig` | FP8 | 16,384 |
| `MxE4M3OnlyGemminiRocketConfig` | FP8 | 16,384 |
| `MxAllGemminiRocketConfig` | FP8, FP4, FP6 | 36,864 |
| `MxE4M3LutGemminiRocketConfig` | FP8, FP4, FP6 | 36,864 |
| `MxE5M2GemminiRocketConfig` | FP8, FP4, FP6 | 36,864 |

The [matrix index](evidence/radiance_selected_mx_profiles_266c593/index.json)
and [regression test](../tests/test_selected_mx_profiles_evidence.py)
tie all 12 cases to their exact profile, source capture, bundle, physical
program, executable, and Spike receipt. FP8 on the FP4-only profile is
rejected before output creation. These are direct numerical qualifications
for the listed mode/shape pairs only. Other legal modes in the multi-mode
profiles and Chipyard wrappers require their own evidence.

For one selected configuration, use the same pinned source, RTL, and RV64
roots as the plain-profile command above:

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxFp4OnlyGemminiRocketConfig.json \
  --case fp4 --out-dir /tmp/mx-fp4-only-profile
```

### Radiance FP8, FP4, and FP6 on DIM8 and DIM32

The source planner and physical command lowerer now derive scratchpad rows,
operand transfers, tile counts, and BF16 readout from the selected square
mesh. The standalone compiler builds the matching `gemmini_dim8` or
`gemmini_dim32` Spike extension from Nicolas's pinned RTL submodule.

Radiance's checked-in `C_out_bf16` bytes come from a DIM16 host model. The
target meshes have different accumulator schedules, so exact comparison to
those bytes would be an invalid qualification. The mesh reference path first
recompiles Radiance's pinned `mx_golden.cpp` for DIM16 and requires a byte-for-byte
match with the source header. It then changes only the mesh dimension, PE tile
extent, and accumulator schedule to derive a target mesh reference from the
**same source operand codes and E8M0 scales**. For FP6, it also unpacks the
source's A/B 64-line LUT banks at the source granularity. The source golden bytes, model
source hashes, transformed model hash, and target reference hash remain in the
bundle provenance. A target mesh reference is labeled separately from source
golden parity throughout the physical receipt and Spike artifact manifest.

Two independent builds per profile matched **20,480 / 20,480 BF16 outputs**
on each of DIM8 and DIM32: FP8 128×128×512 and FP4 64×64×128. All four cases
start from the archived model2MLIR `e9ded36` captures and source checked
Radiance `80f84ca` data. The
[mesh qualification index](evidence/radiance_mx_mesh_reference_266c593/index.json)
and [regression test](../tests/test_radiance_mesh_reference_evidence.py)
pin the profile, generated commands, RV64 ELF, target reference, and Spike
log for each build. This is functional Spike evidence for those shape and
precision pairs. A separate two-run [FP6 qualification index](evidence/radiance_mx_mesh_fp6_reference_266c593/index.json)
and [test](../tests/test_radiance_mesh_fp6_reference_evidence.py) cover
FP6 128×128×2048 on both meshes, comparing another **32,768 / 32,768 BF16
outputs per run**. Across the six selected cases, **73,728** target mesh BF16
outputs match per run. RTL and FPGA execution remain separate gates; the
earlier Nicolas asymmetric matrices independently exercise FP6 modes on
DIM8 and DIM32 using his mesh-specific source goldens.

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim8AllAsymGemminiRocketConfig.json \
  --case fp8 --case fp4 --case fp6 --out-dir /tmp/mx-radiance-dim8
# Repeat with MxDim32AllAsymGemminiRocketConfig and a new output directory.
```

The [31-driver evidence archive](evidence/radiance_mx_gemm_latest_e9ded36_ee22/)
binds the `ee22e0b` Radiance MX GEMM drivers to model2MLIR `e9ded36`, Nicolas's
`gemmini-mx-cleanup` RTL `266c593`, and this compiler's `12cb75d` Spike runs.
model2MLIR captures the PyTorch matmul and its shape; the compiler binds the
drivers' packed operands, scales, LUTs, and output policy from the source
headers to that typed contraction before physical lowering.
All **23 BF16-output** and **8 requantized-output** drivers compiled and matched
their source goldens on Nicolas's pinned Rocket/RoCC Spike extension. The full
comparisons cover **376,832 BF16 values**, **73,728 FP8 codes**, **16,384 FP6
packed bytes**, and **3,328 E8M0 output scales**. The 31 drivers consist of 11
FP4, 7 FP6, and 13 FP8 cases. FP4/FP8 use the MX+VPU profile; FP6 uses the
FP6-only profile because the pinned RTL does not expose FP6+VPU. The archive
contains every model2MLIR capture, MX handoff, bound MLIR, physical command
program, generated issuer, Spike log, and manifest for the first run, plus
receipts for a second independent run. Binary payloads, objects, ELFs, and
extensions are identified by SHA256 in those manifests and can be rebuilt.

The current Radiance `spatter-workloads` revision `80f84ca` has the same 31
driver and generated/committed header hashes as the `ee22e0b` baseline.
Using upstream model2MLIR `e9ded36`, the
[current-source reproduction](evidence/radiance_mx_gemm_80f84ca_upstream_repro_20261010/reproduction.json)
recaptured all 31 contractions and rebuilt all 31 Rocket ELFs on Nicolas's
pinned Spike. The current source generator, source plans, frontend MLIR,
physical programs, objects, ELFs, and Spike logs match the archived baseline.
All 23 BF16 and 8 quantized source comparisons pass. The
[compact archive](evidence/radiance_mx_gemm_80f84ca_upstream_repro_20261010/index.json)
contains current-revision capture and numerical receipts; the matching
generated source and Spike logs remain in the baseline archive. An earlier
independent run also matched every frontend and numerical artifact. This
equivalence is scoped to the 31 byte-identical drivers and headers.

The newer `ucb-bar/radiance-kernels` main revision `82be2c7` still has those
same 31 MX GEMM driver, generator, and materialized-header bytes. The
[new-upstream receipt](evidence/radiance_mx_gemm_upstream_82be2c7_20261010/index.json)
records a fresh full run with model2MLIR `e9ded36` and Nicolas RTL `266c593`:
all 31 PyTorch captures, physical command programs, RV64 ELFs, and stock
Spike outputs match the archived baseline. The guard compared each source
golden, not just the aggregate status. This receipt covers the MX GEMM roster;
upstream's newer nightly HBM, attention, and layernorm workloads require
separate coverage and are not included in the 31-driver claim. Reproduce
the run with `tools.reproduce_radiance_roster` below and
`--compatible-source-revision`.

A [fresh checkout of compiler `4fc4d3a` against Radiance `82be2c7`](evidence/radiance_mx_gemm_fresh_4fc4d3a_82be2c7/index.json)
rebuilt `mx-gemmini-opt` and reran the guarded one-command roster. All 31
model2MLIR captures and Spike comparisons passed, covering **466,944 output
elements** across 23 BF16-output and eight requantized drivers. Every
driver/header, bound MLIR, physical stream, ELF, and Spike-log hash matched
the pinned roster. The compact receipt contains the complete fresh frontend
and Spike indices and source/header census. This checks the published compiler
after the new connected 128³ chain was added.

The [build-selection audit](evidence/radiance_mx_gemm_build_selection_80f84ca.json)
checks all 31 driver hashes against the newer Radiance `80f84ca` tree and
reads its pinned [Makefile](evidence/radiance_mx_gemm_build_selection_80f84ca.Makefile).
It finds **18 named drivers selected by `MU_SRCS`** and **13 source recipes
excluded from that build**. All 31 have compiler-on-Spike numerical receipts;
the receipt count does not mean Radiance built 31 source ELFs. The excluded
recipes include tiles that exceed its 128 KiB scratchpad layout and FP6 cases
without a source data header. Reproduce this audit with
[`tools.audit_radiance_mx_build_selection`](../tools/audit_radiance_mx_build_selection.py)
using the two roster `index.json` files and the selected Radiance checkout.

The two frontend captures produce identical source and profile-bound MLIR. The
first frontend receipts name compiler `2ad71ab`, and the second name
`12cb75d`, because the roster commands were committed between captures. Apart
from that receipt revision field, their capture receipts agree. Both numerical
runs use `12cb75d`; their generated source, objects, ELFs, extensions, and
Spike logs have identical hashes for all 31 cases. Link logs can contain the
output directory and are excluded from the byte-for-byte comparison.

For a fresh checkout, build Radiance's golden tool, then generate **only
missing** driver headers. Radiance commits the FP8 128×128×512 and FP6
128×128×2048 headers; `--all-fp8` would overwrite the former with a different
golden. The roster command preserves committed headers, derives the missing
FP6 sizes from the committed 2048-K data, and checks all driver and header
hashes against the archive. From the `mx-gemmini-mlir` repository root:

```sh
make -C "$RADIANCE_KERNELS_ROOT/lib/golden" mx_golden
python -m tools.materialize_radiance_roster_headers \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --expected-index docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend/index.json \
  --out /new/mx-header-materialization.json
python -m tools.recapture_radiance_roster \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --radiance-opt "$RADIANCE_OPT" \
  --out-dir /new/mx-frontend-roster
python -m tools.requalify_radiance_roster \
  --capture-root /new/mx-frontend-roster \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --out-dir /new/mx-spike-roster --jobs 2
```

For one guarded command that runs all three steps and compares the new MLIR,
physical artifacts, ELFs, and Spike logs with the archived baseline:

```sh
python -m tools.reproduce_radiance_roster \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --radiance-opt "$RADIANCE_OPT" --out-dir /new/mx-reproduction --jobs 2
```

For Radiance `80f84ca` or `82be2c7`, append `--compatible-source-revision` to the
one-command invocation. The flag checks every driver and header hash, source
generator and plan, frontend artifact, physical command stream, ELF, and
Spike log against the baseline before writing a success receipt. Append the
same flag to the separate header-materialization command when using that
revision. Without the flag, the command requires the exact baseline Radiance
revision.

A third run used that command at compiler `66df445`. Its
[reproduction receipt](evidence/radiance_mx_gemm_latest_e9ded36_ee22/one_command_66df445/reproduction.json)
and per-driver receipts show the same captured MLIR, physical source files,
objects, ELFs, extensions, and Spike logs as the archived baseline for all
31 drivers. The default command refuses a changed source/tool revision or
artifact digest before writing a success receipt.

A [fresh GitHub checkout at `205e178`](evidence/mx_fresh_checkout_205e178/index.json)
rebuilt `mx-gemmini-opt` from source, then ran this one-command reproduction
against the current Radiance `80f84ca` checkout. Its 31 fresh frontend and
Spike rows match the archived source, bound MLIR, ELF, and simulator-log
hashes. The same checkout rebuilt and ran the eight-wrapper Spike matrix;
all eight output identities match their archive. The receipt pins the fresh
compiler, model2MLIR, RTL, source, and rebuilt `mx-gemmini-opt` digests.

Repeat the capture and Spike commands with fresh output directories, then run
`tools.archive_radiance_roster` with both pairs and the header materialization
report. It rejects changed frontend artifacts, generated sources, objects,
ELFs, extensions, or Spike logs before making a reviewable archive.

This is full source parity for the current MX **GEMM driver roster** on pinned
Spike. It does not cover the mixed MX+Muon HBM FlashAttention kernel, arbitrary
typed graphs, FPGA execution, or the stock Spike E4M3-direct × E4M3-LUT defect
described above.

### Linkable objects for the Radiance host requantization cases

`tools.compile_object` now lowers each source-bound
`mx_gemmini.host_requantize` graph into one data-free RV64 object entry point.
The entry point runs the typed MX contraction and BF16 readout, then applies
the source-compatible FP8 or FP6 host epilogue using caller-provided output
and scale buffers. FP4 source drivers use the FP8 output-code convention.
The FP6 path also accepts the full 64-pair output LUT as a runtime pointer and
clears packed output storage before writing its two nibbles per byte. The
same host arithmetic generators feed the standalone ELF path; the object
path only changes their storage binding.

The [eight-case object roster](evidence/radiance_host_requant_objects_80f84ca/index.json)
uses the current Radiance `80f84ca` source roster and model2MLIR `e9ded36`
typed graphs. The native MX verifier accepts all eight graphs. Nicolas's
pinned Spike matches **90,112 / 90,112 code bytes** and **3,328 / 3,328
E8M0 scales** against the source goldens. Each object has zero allocated data
bytes and no unresolved symbols. The archived per-case manifests, objects,
ELFs, and Spike logs make the comparison inspectable. This qualifies these
eight host-output shapes and selected profiles on the pinned functional
model; RTL timing and FPGA execution remain separate gates.

From a source-roster reproduction produced as above:

```sh
python -m tools.qualify_host_requant_roster \
  --source-root /new/mx-reproduction/spike \
  --profiles profiles/gemmini-mx-cleanup-266c593 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/mx-host-objects --workers 2
```

## Read-once weight-stationary Radiance MX kernels

The two committed read-once drivers at Radiance `ee22e0b` now use the same
model2MLIR capture, source payload binding, physical command IR, and standalone
Rocket/RoCC compiler as the GEMM roster. The checked
[qualification index](evidence/radiance_ws_read_once_1a7bc9c/index.json)
covers FP8 **256×64×2048** and FP4 down-projection **256×64×5632**. Each ELF
matches all **16,384 BF16 source outputs** on Nicolas's pinned Spike model.
Their physical programs use one M=256 output tile and 32 or 88 K waves. The
checked 16×16-byte weight DMAs cover every packed weight byte exactly once:
131,072 bytes for FP8 and 180,224 bytes for FP4. This matches the source's
read-once weight traffic; the compiler does not split the M tile into four
passes that reload B. The profile is MX+VPU-capable, although these two GEMMs
do not issue VPU operations.

Both `data` files are missing from the clean checkout and are regenerated by
Radiance's pinned `gen_mxgemm_data.py`. The
[fresh materialization receipt](evidence/radiance_ws_read_once_1a7bc9c/fresh_data_materialization.json)
binds the exact driver, generator, and generated byte hashes without replacing
existing data. Two clean compiler runs produced identical captured MLIR,
physical programs, generated issuers, objects, ELFs, extensions, and Spike
logs; their [first](evidence/radiance_ws_read_once_1a7bc9c/index.json) and
[second](evidence/radiance_ws_read_once_1a7bc9c/index_repro.json) indices are
byte identical. Link logs contain the output directory and are excluded from
the stable artifact comparison. The source kernels run through Radiance's
Muon-side MX gateway; this qualification executes equivalent physical commands
through the standalone Rocket/RoCC path and does not establish SoC timing.

From a clean checkout at `ee22e0b`, using the pinned model2MLIR and MXQuant
checkouts, reproduce the two kernels with:

```sh
python -m tools.qualify_radiance_ws_roster \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --radiance-opt "$RADIANCE_OPT" --out-dir /new/mx-ws-run \
  --baseline-index docs/evidence/radiance_ws_read_once_1a7bc9c/index.json
```

### Four-pass re-stream comparison

The committed `gemm_mxgemmini_ws_restream/kernel.cpp` uses four M=64 output
tiles. Its `A_scales_tiled` array is checked as an exact block-contiguous
reordering of the committed canonical E8M0 scales before binding the typed
contraction. A model2MLIR capture of the actual 256×64×2048 source shape
lowers to four output-tile programs, and its standalone ELF matches all
**16,384 BF16 values** in that source's committed golden on pinned Spike. The
[physical trace](evidence/radiance_restream_1df2c5f/physical_program.json.gz)
has four complete reads of B: **128 logical weight-tile reads** and 524,288
transferred weight bytes, versus 32 reads and 131,072 bytes in the read-once
case. Two clean runs have identical
[artifact indices](evidence/radiance_restream_1df2c5f/index.json).

The committed re-stream `data` has a different hash from the regenerated
read-once data. The comparison establishes each source's own numerical parity
and the fourfold weight-traffic difference for their equal shapes; it does not
claim that they used identical operand values or that their cycle counts match
on the Radiance SoC. Reproduce the re-stream case with:

```sh
python -m tools.qualify_radiance_restream \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --radiance-opt "$RADIANCE_OPT" --out-dir /new/mx-restream \
  --baseline-index docs/evidence/radiance_restream_1df2c5f/index.json
```

### Batched decode GEMV projections

The four `gemv_batched_*` Radiance drivers at `ee22e0b` are matrix
projections with M=32, 64, or 128, N=128, and K=2048. The first two use
non-square output tiles. The pinned source generator and `regen_data.sh`
produce each absent `data` blob, including the verification glue. The
[fresh-checkout receipt](evidence/radiance_batched_gemv_552a16b/fresh_data_materialization.json)
checks all four driver and data hashes.

The [qualification index](evidence/radiance_batched_gemv_552a16b/index.json)
records model2MLIR capture, typed payload binding, physical command lowering,
standalone Rocket/RoCC ELF emission, and Nicolas Spike comparison. FP8 M32,
M64, and M128 match **4,096**, **8,192**, and **16,384** source BF16 outputs;
FP4 M128 matches **16,384**. Each one-tile program uses 16 K waves and loads
each packed weight byte exactly once: 262,144 bytes for FP8 or 131,072 bytes
for FP4. The test runs on Nicolas's MX+VPU standalone profile; these four
matrix programs do not issue VPU operations. The PyTorch capture establishes
the operation and shape, while the actual quantized input bytes, scales, and
goldens come from the pinned Radiance generator.

Two runs from the committed compiler revision produce identical captured
MLIR, physical commands, generated issuer, objects, ELFs, and Spike logs.
Link logs contain output paths and are excluded from that stable comparison.
From a clean Radiance checkout at `ee22e0b`, reproduce with:

```sh
python -m tools.qualify_radiance_batched_gemv_roster \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --radiance-opt "$RADIANCE_OPT" --out-dir /new/mx-batched-gemv \
  --baseline-index docs/evidence/radiance_batched_gemv_552a16b/index.json
```

This qualifies the exact batched source kernels and physical traffic on Spike.
It does not establish mixed MX/Muon scheduling or FPGA cycle counts.

### GQA QK numerical domain and source-derived candidate

The pinned `flash_attention_mx_gqa` source generator emits FP8 Q and K codes
for a 64×64×64 first QK tile. Its Python matrix golden uses BF16 product and
accumulator precision. Nicolas's DIM16 MX path instead uses an E4M3 product
and a reduced-precision accumulator ramp before applying E8M0 scales. For
the exact generated header at Radiance `ee22e0b`, **244,154 of 262,144** raw
Q×K products exceed E4M3's maximum finite magnitude, 448. A direct compiler
probe with those unchanged bytes returned 4,096 BF16 mismatches on pinned
Spike; its first reported results were NaNs. The generator's final attention
golden therefore cannot establish parity for that hardware path.

The compiler now accepts a **separate, explicitly labeled candidate**: divide
the source Q and K values by 64 before FP8 encoding, and increase each E8M0
scale exponent by six. The resulting source-derived QK tile is finite under
Nicolas's product and accumulator precision. A model2MLIR `matmul` capture
binds those candidate bytes, and the generated Rocket/RoCC program matches
all **4,096 BF16 outputs** of the hardware-aware candidate oracle on Spike.
The [candidate index](evidence/radiance_gqa_qk_candidate_5baebbe/index.json)
records source, policy, compiled artifacts, and Spike identities; its
[reproduction index](evidence/radiance_gqa_qk_candidate_5baebbe/index_repro.json)
is byte identical. The payload origin and derivation are checked by the MLIR
and bundle verifiers, so this candidate cannot be presented as unchanged
source-header parity.

From a clean checkout at `ee22e0b` with its pinned `lib/mxgemmini` submodule,
reproduce the candidate with:

```sh
python -m tools.qualify_radiance_gqa_qk \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --scale-shift 6 \
  --out-dir /new/mx-gqa-qk \
  --baseline-index docs/evidence/radiance_gqa_qk_candidate_5baebbe/index.json
```

This proves only one source-derived QK tile. The unchanged GQA source data,
causal masking, online softmax, requantization, PV, multi-head scheduling,
Muon handoff, and final attention output remain unqualified. The next source
step is to revise and verify the Radiance generator's numerical domain before
claiming exact source parity.

#### Regenerated GQA fixture against Nicolas's Spike model

The [experimental source patch](patches/radiance_gqa_hardware_model.patch)
adds an explicit `--mx-hardware-shift 6` option to the Radiance GQA data
generator. It rounds the shifted operands to E4M3, increases the E8M0 scales,
and computes the full GQA golden with Nicolas's E4M3 product and accumulator
schedule. The ordinary generator invocation retains its old behavior. The
qualification tool applies this patch to a **copy** of the source files and
checks the resulting generator, model, header, and patch digests.

Using the current `spatter-workloads` source at `80f84ca` and its pinned
`lib/mxgemmini` submodule, the regenerated GQA header has no raw Q×K products
above E4M3's finite range in its first tile. All **32,768** generated `O_gold`
BF16 codes are finite and agree between the header and `.npy` export; the full
golden has **8.33%** relative error against the FP32 reference.
The first source-generated QK tile is captured from PyTorch `matmul` by
model2MLIR, bound to those emitted bytes, compiled to physical MX commands and
a standalone Rocket/RoCC ELF, and checked on Nicolas's pinned Spike extension:
**4,096 / 4,096 BF16 outputs match**. A clean `ee22e0b` checkout and the
current `80f84ca` source produced the same generated header, physical program,
and ELF hashes; their [indexes](evidence/radiance_gqa_generated_80f84ca/index_prior_source.json)
differ only in the source revision. The [qualification index](evidence/radiance_gqa_generated_80f84ca/index.json)
records the exact tool, source, payload, object, and Spike digests.
It was generated from compiler commit `abad502`; check out that commit when
comparing the complete index byte for byte, since the index records `HEAD`.

Reproduce from a clean `80f84ca` Radiance checkout with its `lib/mxgemmini`
submodule initialized:

```sh
python -m tools.qualify_radiance_gqa_qk \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --scale-shift 6 \
  --generated-hardware-model --out-dir /new/mx-gqa-generated
```

This is an experimental source correction and a **first QK tile** result.
The patched generator has not been merged into `radiance-kernels`; the full
mixed MX/Muon attention kernel, its Muon softmax and requantization, PV tiles,
causal masking, and final `O_gold` comparison still need execution evidence.

#### All GQA QK heads and used key blocks

The [QK roster qualification](evidence/radiance_gqa_qk_roster_80f84ca/index.json)
uses the same regenerated header and model2MLIR capture. It follows the
source driver's `h / FA_GRP` mapping for **8 query heads × 2 used key blocks**,
checks each tile against the patched source model, and compiles each contraction
to a separate standalone RoCC ELF. All **65,536 / 65,536 BF16 outputs** match
Nicolas's Spike model. Each physical program has 83 steps. The selected source
arrays contain eight distinct Q tiles and four distinct K tiles, as required
by the four-query-heads-per-KV-head mapping.
The archived roster was built from compiler commit `38de229`; its two runs
have identical indexes, payloads, physical programs, ELFs, and Spike logs.

After running the generated fixture command above, run:

```sh
python -m tools.qualify_radiance_gqa_qk_roster \
  --prepared-dir /new/mx-gqa-generated --source-root "$RADIANCE_KERNELS_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-gqa-qk-roster --jobs 4
```

This covers the MX QK contractions for the source's used blocks. The causal
mask, Muon online softmax and P requantization, MX PV contractions, and final
attention output are separate stages and remain to be qualified together.

#### Linkable QK object with runtime buffers

The [runtime object receipt](evidence/radiance_gqa_runtime_object_80f84ca/two_payload_spike/index.json)
binds the typed head 0, block 0 QK program to Nicolas's Rocket profile and
emits `mx_issue.o` with six pointer arguments: activation codes and scales,
BF16 output, scale scratch, and weight codes and scales. The object defines
only `mx_issue`, imports no symbols, and has zero allocated data-section bytes.
The head 7, block 1 bound program emits the **same object SHA-256**
(`1c70015f1b8a7bdd7a53b8a6498e5dfc220fa2046dbead3ea57ec2532f4b26ab`)
despite distinct Q and K payloads. One RV64 ELF links that object once and
calls it twice with distinct runtime pointers. Nicolas's stock Spike reports
**0 / 8,192 BF16 mismatches**, and an independent build matches the receipt's
ELF, extension, and Spike log hashes. The [object manifest](evidence/radiance_gqa_runtime_object_80f84ca/head0_object/object_manifest.json),
[generated command issuer](evidence/radiance_gqa_runtime_object_80f84ca/head0_object/mx_issue.c),
and [linked ELF](evidence/radiance_gqa_runtime_object_80f84ca/two_payload_spike/mx_runtime_qk.elf)
are archived alongside the source bundles in the QK roster above.

From this repository checkout, with the pinned Gemmini RTL checkout and
initialized `software/libgemmini` and `software/gemmini-rocc-tests` submodules:

```sh
export PYTHONPATH="$MODEL2MLIR_ROOT:$MXQUANT_ROOT"
QK=docs/evidence/radiance_gqa_qk_roster_80f84ca
python -m tools.emit_mx_object \
  --mlir "$QK/head0_block0/payload_bound.mlir" \
  --bundle "$QK/head0_block0/bundle" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --out-dir /new/mx-qk-object
python -m tools.qualify_runtime_qk_object \
  --object-dir /new/mx-qk-object \
  --first-bundle "$QK/head0_block0/bundle" \
  --second-bundle "$QK/head7_block1/bundle" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --out-dir /new/mx-qk-runtime \
  --baseline-index docs/evidence/radiance_gqa_runtime_object_80f84ca/two_payload_spike/index.json
```

This qualifies pointer rebinding for two source-derived MX QK contractions.
The source fixture still uses its documented Q/K overflow correction. It does
not execute Muon or establish a shared-memory PV handoff; those require the
source Muon requantization and ordering work described below.

The [PV handoff diagnostic](evidence/radiance_gqa_pv_handoff_80f84ca/diagnostic.json)
pinpoints the next numerical mismatch. Given the **same BF16 P values** for
head 0, block 0, the experimental Python golden's general MX quantizer and a
translation of the source Muon requantizer differ on **4,093 / 4,096 P codes**
and **128 / 128 P scales**. The translation's E4M3 encoder matches the source
C helper for 228,480 nonnegative BF16/scale inputs; its scale selector uses
rounded `torch.exp` as a proxy for Muon's `mu_fexp`, so the counts are a
diagnostic rather than an executed Muon result. The source-style P and the
generated V operand have a raw E4M3 PV product bound of 15 in this probe.
The generated `O_gold` is therefore an internal model output, **not yet a
golden for the existing mixed MX/Muon kernel**.

#### First PV contraction with a Muon P proxy

The [PV proxy qualification](evidence/radiance_gqa_pv_proxy_80f84ca/index.json)
starts with the qualified head 0, block 0 BF16 QK tile. The first key block is
fully visible to all query rows. It applies the pinned Muon P scale and E4M3
encoding formulas, using BF16-rounded Torch `exp` in place of `mu_fexp`, and
takes V codes and scales from the same generated Radiance header. A fresh
PyTorch `matmul` capture goes through model2MLIR, typed MX source-resource
binding, physical command lowering, and RV64 emission. Nicolas's stock Spike
matches **4,096 / 4,096 BF16 PV outputs** against its reduced-precision MX
numerical model. Two independent builds have identical qualification indexes.
The [linkable PV object](evidence/radiance_gqa_pv_proxy_80f84ca/linkable_object/object_manifest.json)
is byte identical to the QK object above, with every operand supplied by a
pointer. The [typed PV program](evidence/radiance_gqa_pv_proxy_80f84ca/payload_bound.mlir),
[source-bound bytes](evidence/radiance_gqa_pv_proxy_80f84ca/bundle/manifest.json),
and [Spike ELF](evidence/radiance_gqa_pv_proxy_80f84ca/build/mx_program.elf)
are archived.

```sh
export PYTHONPATH="$MODEL2MLIR_ROOT:$MXQUANT_ROOT"
python -m tools.qualify_radiance_gqa_pv_proxy \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/mx-gqa-pv-proxy \
  --baseline-index docs/evidence/radiance_gqa_pv_proxy_80f84ca/index.json
```

This is an MX-side PV qualification for **proxy P bytes**. The simulator run
does not execute Muon, prove `mu_fexp` bit parity, or exercise the tiled
Muon-to-MX shared-memory handoff. The experimental Q/K headroom correction
also remains part of the fixture. The local Cyclotron Muon model implements
`fexp.h` as rounded host `exp`, while the Muon RTL FPEX uses a fixed-point LUT
and interpolation. The proxy is therefore useful for the MX consumer test,
but RTL P-code parity needs a separate SFU check or device trace.

#### Executed Muon-to-MX PV cross-check on Cyclotron

The [Cyclotron cross-check](evidence/radiance_gqa_cyclotron_pv_80f84ca/index.json)
builds the **unmodified** GQA source kernel and a copied diagnostic version
against the archived header. Both execute the complete eight-head, two-block
kernel with Cyclotron's MX functional co-model enabled. The diagnostic stores
the first P tile and its scales after Muon requantization, then the first BF16
PV tile after MX compute. It leaves the source math and MX commands intact;
the final 32,768-value BF16 O buffer is byte identical between the two runs.
The [probe patch](evidence/radiance_gqa_cyclotron_pv_80f84ca/probe.patch)
and both built RV32 ELFs are archived with the receipt.

Muon's tiled scratchpad P yields exactly the same **4,096 E4M3 codes** and
**128 E8M0 scales** as the compiler's source-derived PV operand after
untilling. Cyclotron's MX PV result matches **4,096 / 4,096 BF16 values** of
the compiler's Nicolas-Spike PV golden, byte for byte. The archived
[P scratch dump](evidence/radiance_gqa_cyclotron_pv_80f84ca/p_scratch.bin.gz),
[PV tile](evidence/radiance_gqa_cyclotron_pv_80f84ca/pv_bf16.bin), and
[unmodified final O](evidence/radiance_gqa_cyclotron_pv_80f84ca/baseline_o.bin.gz)
support the comparisons. A second isolated build produced the same receipt.

```sh
python -m tools.qualify_radiance_gqa_cyclotron_pv \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --radiance-lib-root "$RADIANCE_BUILT_LIB_ROOT" \
  --cyclotron-root "$CYCLOTRON_ROOT" --llvm-muon "$LLVM_MUON_ROOT" \
  --riscv-root "$RISCV_ROOT" --out-dir /new/mx-gqa-cyclotron-pv \
  --baseline-index docs/evidence/radiance_gqa_cyclotron_pv_80f84ca/index.json
```

This qualifies one actual Muon-to-MX cutpoint in Cyclotron's functional
models. It does not establish RTL FPEX parity, FPGA behavior, or final
attention parity with the generator's `O_gold`. The compiler still needs a
mixed Muon/MX lowering and runtime buffer handoff to emit the whole attention
kernel itself. The first-block Q/K fixture also retains its explicit
headroom correction for Nicolas's reduced-precision MX path.

```sh
python -m tools.diagnose_radiance_gqa_pv_handoff \
  --prepared-dir /new/mx-gqa-generated --source-root "$RADIANCE_KERNELS_ROOT" \
  --out-json /new/gqa-pv-handoff.json
```

To compile the actual PV and final output, the compiler needs runtime P codes
and scales produced by Muon, an ordered Muon-to-MX scratchpad handoff, and a
source-faithful numerical oracle for the full output. This kernel's softmax
runs on Muon; a separate MX VPU softmax test cannot stand in for it.

#### All 16 executed GQA PV cutpoints on Nicolas Spike

The first-block test exposed an accumulation defect in the local Cyclotron MX
co-model: it added each new matmul to the prior C scratchpad contents even when
the source command set `ex_accumulate=0`. Nicolas's
`gemmini-mx-cleanup` Spike model explicitly clears C for that command. The
[isolated Cyclotron correction](evidence/radiance_gqa_pv_roster_80f84ca/cyclotron_overwrite.patch)
implements the same overwrite rule. It is an evidence patch applied to a
disposable Cyclotron worktree; the main simulator checkout is unchanged. All
35 MX simulator tests pass with the patch, including a renewed overwrite and
explicit-accumulation check in the
[test log](evidence/radiance_gqa_pv_roster_80f84ca/cyclotron_model_tests.log).

The [16-tile receipt](evidence/radiance_gqa_pv_roster_80f84ca/index.json)
builds the pinned Radiance GQA kernel, executes all eight query heads and two
key blocks with the corrected Cyclotron model, and captures each Muon P tile,
its scales, and the resulting MX PV tile. Each PV tile agrees with Nicolas's
reduced-precision numerical model for **4,096 / 4,096 BF16 values**. A single
compiler emitted RV64 MX object, originally lowered from the model2MLIR PV
capture, then accepts all 16 distinct runtime P/V payloads on Nicolas's stock
Rocket/RoCC Spike extension. Spike matches **65,536 / 65,536 BF16 values**.
The probe leaves the final source O buffer byte identical to the unmodified
kernel running on the corrected model.

```sh
git -C "$CYCLOTRON_ROOT" worktree add --detach /new/cyclotron-mx-probe 2d6adad
git -C /new/cyclotron-mx-probe apply --unidiff-zero \
  "$MX_MLIR_ROOT/docs/evidence/radiance_gqa_pv_roster_80f84ca/cyclotron_overwrite.patch"
(cd /new/cyclotron-mx-probe && cargo build --release)

"$MODEL2MLIR_PYTHON" -m tools.qualify_radiance_gqa_pv_roster \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --radiance-lib-root "$RADIANCE_BUILT_LIB_ROOT" \
  --cyclotron-root /new/cyclotron-mx-probe --llvm-muon "$LLVM_MUON_ROOT" \
  --riscv-root "$RISCV_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --out-dir /new/mx-gqa-pv-roster \
  --baseline-index docs/evidence/radiance_gqa_pv_roster_80f84ca/index.json
```

The archived [Muon P](evidence/radiance_gqa_pv_roster_80f84ca/p_tiles.bin.gz),
[P scales](evidence/radiance_gqa_pv_roster_80f84ca/p_scales.bin.gz),
[MX PV](evidence/radiance_gqa_pv_roster_80f84ca/pv_tiles.bin.gz),
[runtime payloads](evidence/radiance_gqa_pv_roster_80f84ca/runtime_payloads.tar.gz),
[Spike ELF](evidence/radiance_gqa_pv_roster_80f84ca/mx_runtime_pv_roster.elf.gz),
and [Spike log](evidence/radiance_gqa_pv_roster_80f84ca/spike.log)
pin this result. This qualifies the **MX portions** of the source kernel on
functional simulators. The compiler does not yet emit the whole mixed Muon/MX
kernel, and neither simulator proves RTL FPEX or FPGA behavior. The corrected
Cyclotron final O still differs from the generator's `O_gold` in 31,852 of
32,768 BF16 values, so that generated array remains unsuitable as a full
attention parity oracle.

#### Executed final GQA recurrence

The [final recurrence qualification](evidence/radiance_gqa_final_recurrence_80f84ca/index.json)
captures the source Muon kernel's 64 BF16 row corrections and running
denominators after each of the 16 PV tiles. It combines those row states with
the independently qualified, executed MX PV tiles above. The source
`rescale_accumulate` and `finalize_O` formulas, including BF16 rounding of the
accumulator and reciprocal, reconstruct **32,768 / 32,768 BF16 final O values**
exactly. A separate run of the diagnostic kernel produces the same final O as
the unmodified source kernel on the corrected Cyclotron model. The
[row-state capture](evidence/radiance_gqa_final_recurrence_80f84ca/corr_l.bin.gz),
[reconstructed O](evidence/radiance_gqa_final_recurrence_80f84ca/reconstructed_o.bin.gz),
[probe patch](evidence/radiance_gqa_final_recurrence_80f84ca/probe.patch),
and [RV32 ELF](evidence/radiance_gqa_final_recurrence_80f84ca/probe.elf.gz)
are archived.

```sh
"$MODEL2MLIR_PYTHON" -m tools.qualify_radiance_gqa_final_recurrence \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --radiance-lib-root "$RADIANCE_BUILT_LIB_ROOT" \
  --cyclotron-root /new/cyclotron-mx-probe --llvm-muon "$LLVM_MUON_ROOT" \
  --riscv-root "$RISCV_ROOT" --out-dir /new/mx-gqa-final-recurrence \
  --baseline-index docs/evidence/radiance_gqa_final_recurrence_80f84ca/index.json
```

This establishes a byte-exact **functional-model final-output oracle** for the
current source kernel. The compiler still does not emit the Muon softmax,
online accumulation, and finalization around its MX commands. The generated
`O_gold` remains 31,852 BF16 values away from this executed result, and RTL
FPEX/FPGA parity remains a separate gate.

## Isolated weight-LUT Spike correction across all legal modes

Nicolas's pinned MxGen submodule has a
[mode-9 Chisel test](evidence/nicolas_e4m3_lut_pe_266c593/receipt.json)
for E4M3 quad arithmetic. Its harness forces `lut_en` and checks 300
randomized two-activation × two-weight trials, or 1,200 BF16 lane results.
The [saved test log](evidence/nicolas_e4m3_lut_pe_266c593/test.log) passes on
MxGen `dba3e7e` under Gemmini `266c593`. The receipt also pins the
`ExecuteController.scala` source that includes `weight_lut_en` in output-column
packing. This is PE arithmetic evidence; the harness does not simulate LUT
DMA, the ExecuteController, the full RoCC loop, or FPGA execution. Reproduce
it with [`tools.qualify_nicolas_e4m3_lut_pe`](../tools/qualify_nicolas_e4m3_lut_pe.py).

The [candidate qualification index](evidence/nicolas_spike_weight_lut_candidate_all_modes_266c593/qualification.json)
records a full rerun with the existing
[weight-LUT lane patch](evidence/nicolas_generated_modes_266c593/spike_weight_lut_quad_candidate.patch)
applied only to an isolated `software/libgemmini` worktree. Its parent Gemmini
RTL and profiles remain at Nicolas's `266c593`, while the source-built Spike
extension differs from stock. Latest model2MLIR `e9ded36` captured the
operations, and compiler `0d7e31b` generated the physical programs and RV64
ELFs. The archived receipts cover **36 / 36 legal modes on each of DIM8,
DIM16, and DIM32**, comparing **442,368 / 442,368 BF16 outputs** across 108
programs. DIM16 combines 35 Nicolas source/generated tests with the Radiance
direct E4M3×E4M3 driver; DIM8 and DIM32 use 36 Nicolas tests each.

For **105 modes**, the patched run reproduces a stock-passing ELF and Spike
output log byte for byte. The remaining **three** are direct E4M3 activation
× E4M3 LUT weight, one per mesh: each keeps the identical stock-failing ELF
and passes its full 4,096-value source golden only on the patched extension.
The [archive command](../tools/archive_spike_lut_candidate.py) checks every
comparison against a pinned stock receipt and preserves the candidate receipt
and log. Nicolas's RTL `ExecuteController.mx_multi_elem` includes
`weight_lut_en` in weight-column packing; the isolated patch adds the
corresponding Spike state check. This experiment does not merge the patch or
qualify the three cells on stock Spike, RTL simulation, or FPGA.

## Alternating FP6 scale halves on Nicolas's Spike

The [reproducible qualifier](../tools/qualify_fp6_alternating_spike.py) compiles
the source-bound Radiance FP6 128×128×1024 generated-header fixture with K
tile 512 and
`rtl_alternating` physical scheduling. It copies Nicolas's pinned
`software/libgemmini` into an isolated directory and applies the
[two-line selector correction](evidence/fp6_alternating_spike_266c593/spike_fp6_scale_selector.patch)
only there. The RTL `ExecuteController.scala` takes activation and weight
scale selectors from command bits 60 and 61; `ScaleFactorMem.scala` uses those
selectors to choose the read banks. Stock Spike already applies the selectors
in its other MX compute paths, but the FP6 LUT path read half zero.

The [receipt](evidence/fp6_alternating_spike_266c593/index.json) records the
same RV64 ELF under both extensions: **16,362 / 16,384 BF16 mismatches on stock
Spike**, and **0 / 16,384 on the isolated corrected extension**. The
[stock](evidence/fp6_alternating_spike_266c593/stock_spike.log) and
[corrected](evidence/fp6_alternating_spike_266c593/corrected_spike.log) logs
are archived with source, physical-program, ELF, and extension hashes. The
compiler marks this run `source_golden_matched_on_experimental_spike`; it does
not make the pinned Spike or RTL path qualified.

A [scoped production RTL module probe](evidence/nicolas_scale_mem_rtl_266c593/README.md)
now compiles Nicolas's unchanged `ScaleFactorMem.scala` in isolation. It
checks all four activation/weight half combinations at DIM16 with reset
between cases and complete row-cycle alternation at a four-lane geometry.
The observed E8M0 sums are 11, 14, 41, and 44, with 11 again after returning
to half zero. The archived [test](../tests/test_nicolas_scale_mem_rtl_evidence.py)
also checks that the compiler's 16-wave FP6 stream encodes alternating
`CONFIG_SCALE_MEM` selector bits `00` and `11`. ExecuteController, DMA timing,
the full RTL contraction, and FPGA execution still need separate tests.

Reproduce from a checkout with Nicolas's `266c593` RTL and its pinned
`software/libgemmini` and `software/gemmini-rocc-tests` submodules:

```sh
python -m tools.qualify_fp6_alternating_spike \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp6-alternating-spike
```

The [checked-in Radiance FP6 128×128×2048 driver](evidence/radiance_fp6_alternating_80f84ca/index.json)
extends this test to all **16 K waves**. Its driver and header at Radiance
`80f84ca` have the same hashes as the earlier source-qualified serial case.
The model2MLIR `e9ded36` profile-bound capture is rebound to those exact
source payload bytes; the archived [physical program](evidence/radiance_fp6_alternating_80f84ca/physical_program.json)
contains 1,289 commands with alternating scale uploads and selections. The
1,340 physical steps differ from the archived serial schedule in exactly
24 places: two scale uploads and one selector command in each odd K wave.
The compute, LUT, operand-transfer, and readout steps are identical. The
same compiler ELF has **16,368 / 16,384 BF16 mismatches on pinned Spike** and
**0 / 16,384 on the isolated corrected Spike extension**. Two fresh runs
produced identical source, command, ELF, extension, and output-log hashes.
The [issuer](evidence/radiance_fp6_alternating_80f84ca/mx_issue.c),
[bound MLIR](evidence/radiance_fp6_alternating_80f84ca/payload_bound.mlir),
[stock log](evidence/radiance_fp6_alternating_80f84ca/stock_spike.log), and
[corrected log](evidence/radiance_fp6_alternating_80f84ca/corrected_spike.log)
are archived. Reproduce with:

```sh
python -m tools.qualify_fp6_alternating_spike \
  --rtl-root /path/to/gemmini-mx-cleanup \
  --riscv-root /path/to/riscv-tools \
  --radiance-root /path/to/radiance-kernels \
  --out-dir /tmp/radiance-fp6-alternating-spike
```

This is a numerical check against an experimental Spike correction. The
16-wave scale path still needs RTL or FPGA execution before it can be marked
hardware-qualified.

## Four-tile FP8 GEMM with tilewise VPU epilogue

The [qualifier](../tools/qualify_radiance_tilewise_vpu_x2.py) captures
`torch.matmul(lhs, rhs) * 2.0` with model2MLIR `e9ded36` and the external MX
quantization adapter. It requires a complete original graph containing one
matmul followed by multiplication by the constant 2.0, one quantized FP8 MX
site, and no opaque operation. The contraction's packed A/B codes, E8M0
scales, and BF16 reference come from the pinned Radiance
`mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp` driver and its generated
header at `80f84ca`. Radiance excludes this `tk256` driver from its 128 KiB
build because the C tile does not fit beside double-buffered operands. The
selected Nicolas MX+VPU profile has 256 KiB, so this is a compiler execution
against a source-generated golden, not parity with a source-built ELF. The
original source driver describes GEMM alone. The ×2
epilogue is taken from the captured PyTorch graph; its expected BF16 values
are derived from the source GEMM golden using exact BF16 multiplication.

The handoff currently exports the matmul site only. The checked payload binder
adds a typed, SSA-connected `mx_gemmini.vpu_execute` between the contraction
and BF16 readout, with the explicit module policy
`bf16_muls_x2_each_output_tile_v1`. The physical lowerer applies it to the
reused C scratchpad tile after each tile's final K wave and before its BF16
readout. It rejects missing or changed policy, scratchpad row, operation, or
immediate. This is a narrow, source-bound composition rule for the selected
DIM16 MX+VPU profile; a general graph-level epilogue lowering and scratchpad
lifetime planner are still needed.

The [two-run receipt](evidence/radiance_tilewise_vpu_x2_266c593/index.json)
and [regression test](../tests/test_radiance_tilewise_vpu_x2_evidence.py)
record four output tiles, four compiler-issued VPU commands, and
**65,536 / 65,536 matching BF16 values** on Nicolas's pinned Spike. The
[bound MLIR](evidence/radiance_tilewise_vpu_x2_266c593/tilewise_bound.mlir),
[physical commands](evidence/radiance_tilewise_vpu_x2_266c593/build/physical_program.json.gz),
and [Spike log](evidence/radiance_tilewise_vpu_x2_266c593/build/spike.log)
are archived. The two runs reproduce all source, MLIR, command, object, ELF,
extension, and numerical hashes. The raw artifact manifests differ only in
the link-log hash because the linker warning embeds the output directory.

A [fresh current-compiler rerun](evidence/radiance_mx_vpu_80f84ca_upstream_repro_20261010/README.md)
repeated both this FP8 case and the generated FP4 case below with Radiance
`80f84ca`, upstream model2MLIR `e9ded36`, Nicolas RTL/Spike `266c593`, and
compiler `d548fd7`. Each again matched all 65,536 BF16 values. The frontend,
bound MLIR, physical stream, generated source, ELF, extension, and Spike log
hashes are identical to the original archives; only compiler provenance and
the linker log digest changed. The [receipt check](../tests/test_current_radiance_vpu_reproduction.py)
keeps the FP4 derived-fixture scope explicit.

Reproduce from a checkout with the pinned Radiance header materialized by
`gen_mxgemm_data.py fp8 256 256 256` and its `mx_golden` helper built:

```sh
python -m tools.qualify_radiance_tilewise_vpu_x2 \
  --model2mlir-root /path/to/model2MLIR-e9ded36 \
  --mxq-root /path/to/microscaling-quant-b4af543 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /tmp/radiance-tilewise-vpu-x2
```

## Captured BF16 scalar MULS across four MX output tiles

The same four-tile FP8 source matrix can now take a finite BF16 scalar from a
captured PyTorch `torch.matmul(lhs, rhs) * scalar` graph. The selected scalar is
checked against model2MLIR's original trace and typed frontend MLIR. The binder
records its 16-bit value on the `mx_gemmini.vpu_execute` operation and as a
module policy attribute; physical lowering rejects a mismatch. It issues one
in-place MULS command after the final K wave of each output tile and before
readout. `tools.qualify_source_mx` exposes the same lowering with
`--tilewise-vpu-muls-bf16-bits`.

For scalar **1.5** (`0x3fc0`), the [archived capture and two-run
receipt](evidence/radiance_tilewise_vpu_scalar_266c593/index.json) use current
model2MLIR `e9ded36`, Radiance source/header `80f84ca`, compiler `082c47e`,
and Nicolas RTL/Spike `266c593`. Both runs matched **65,536 / 65,536 BF16
values** on Nicolas's pinned Spike, including the four VPU commands. The
[regression test](../tests/test_tilewise_vpu_scalar_evidence.py) rechecks the
captured graph, source bytes, MLIR binding, physical stream, generated C,
derived reference, and simulator receipts. A host comparison also matched
Nicolas's `vpu_ref.h` multiplication for all 65,280 finite BF16 inputs for each
of six scalar values (`0x3f00`, `0x3fc0`, `0x4000`, `0xbf80`, `0x0001`,
`0x7f7f`). The ×2 path retains its existing bytes and receipts.

The Radiance C driver supplies the GEMM operands and BF16 matrix golden; its
source does not contain the scalar epilogue. The 1.5 result is therefore parity
with the captured graph's derived BF16 output on the selected source operands,
not parity with a source-built Radiance ELF. The binder recognizes scalar
MULS and ADDS epilogues after one contraction, including the ordered chain
qualified below. General graph-level epilogue
lowering and scratchpad lifetime planning remain open.

Reproduce with the same pinned checkouts and generated source header as the ×2
case above:

```sh
python -m tools.qualify_radiance_tilewise_vpu_x2 \
  --model2mlir-root /path/to/model2MLIR-e9ded36 \
  --mxq-root /path/to/microscaling-quant-b4af543 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --scalar-bits 0x3fc0 \
  --out-dir /tmp/radiance-tilewise-vpu-scalar-1p5
```

The general source compiler can also take the capture sidecars directly:

```sh
python -m tools.qualify_source_mx \
  --mlir /path/to/profile_bound.mlir \
  --driver /path/to/mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/mx-captured-scalar \
  --tilewise-vpu-from-capture \
  --capture-trace /path/to/capture_trace.json \
  --capture-quantization-manifest /path/to/quantization_manifest.json \
  --capture-frontend-mlir /path/to/frontend.mlir
```

The [capture adapter](../mx_gemmini_support/capture_epilogue.py) derives the
scalar from the original graph, checks that graph and the frontend MLIR against
the digest-gated handoff, and rejects scalars that cannot be represented
exactly as finite BF16 immediates. The [two CLI receipts](evidence/radiance_tilewise_vpu_scalar_266c593/cli_capture_index.json)
from compiler `b5637a3` reproduce the previously archived bound MLIR, source
bundle, physical program, generated issuer, ELF, Spike log, and all 65,536
outputs. This is a supported narrow graph pattern; the general source CLI
still requires an explicit source payload and selected target profile.

## Captured BF16 scalar ADDS across four MX output tiles

The capture adapter also recognizes a single `torch.matmul(lhs, rhs) + scalar`
graph and lowers the captured addition to one in-place VPU `ADDS` command per
output tile. It verifies the original `aten.add.Tensor` node, its scalar and
empty keyword arguments, the model2MLIR handoff digest, and the BF16 immediate
before binding the typed MX operations. A nondefault `alpha` is rejected.
The physical lowerer checks the scalar against the VPU command and derives
the full BF16 output with Nicolas's round-to-nearest-even scalar arithmetic.

For **1.5** (`0x3fc0`), two independent builds of each FP8 and FP4 program
from compiler `8478ac2` matched **65,536 / 65,536 BF16 outputs** on Nicolas's
RTL `266c593` pinned Spike extension. The [FP8 evidence](evidence/radiance_tilewise_vpu_adds_266c593/fp8/index.json),
[FP4 evidence](evidence/radiance_tilewise_vpu_adds_266c593/fp4/index.json), and
[regression test](../tests/test_tilewise_vpu_adds_evidence.py) connect the
PyTorch capture, typed MLIR, bound payload, four physical ADDS commands,
generated C, ELF, and simulator receipts. An independent host comparison
matched `vpu_ref.h` for every **65,280 finite BF16 input code** with each of
five scalars: `0x3fc0`, `0x3f00`, `0xbf80`, `0x0001`, and `0x7f7f`.
The [general source CLI receipt](evidence/radiance_tilewise_vpu_adds_266c593/fp8/cli_capture_index.json)
records two more capture-driven FP8 Spike builds; their MLIR, bundle,
physical commands, C issuer, ELF, Spike log, and full output match the FP8
qualifier. The only receipt difference is the path-bearing link log hash.

The FP8 GEMM operands and matrix golden come from Radiance source `80f84ca`;
the addition exists in the captured PyTorch graph, not in that C driver.
The FP4 256×256 fixture is generated from pinned Radiance code and has no
committed matching C driver. These results establish source-derived numerical
parity for this graph pattern, not source ELF parity, arbitrary epilogues,
or a qualified FPGA bitstream. Nicolas's checked-in FP4+VPU profile includes
direct FP8 and FP4 modes. His separate FP6 LUT profile has no VPU, so this
work does not imply an FP6+VPU hardware configuration.

With the same pinned inputs shown in the scalar MULS example, reproduce the
selected addition by adding `--epilogue adds`:

```sh
python -m tools.qualify_radiance_tilewise_vpu_x2 \
  --model2mlir-root /path/to/model2MLIR-e9ded36 \
  --mxq-root /path/to/microscaling-quant-b4af543 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --epilogue adds --scalar-bits 0x3fc0 \
  --out-dir /tmp/radiance-tilewise-vpu-adds-1p5
```

The generated FP4 fixture uses the same options with
`tools.qualify_radiance_fp4_derived_tilewise_vpu_x2`. The general
`tools.qualify_source_mx --tilewise-vpu-from-capture` invocation above selects
ADDS directly from its supplied capture sidecars.

## Ordered scalar MX VPU chain from a PyTorch capture

The capture adapter accepts up to 16 digest-checked finite, exactly
representable BF16 `aten.mul.Tensor` and `aten.add.Tensor` literals after one
matmul. Its typed MX result is a single SSA chain: contraction, ordered
in-place VPU operations, then BF16 readout. Physical lowering checks the
sequence against a module policy and every command immediate, keeps the
accumulator tile live across the operations, and repeats the chain after each
output tile's final K wave. It derives the expected output by rounding to
BF16 **after each operation**, matching the selected VPU model. An unrounded
PyTorch FP32 intermediate has different numerical semantics.

For captured `matmul * 2.0 + 1.5`, compiler `3935032` produced eight VPU
commands across four output tiles for both source-bound FP8 and generated
FP4. Two independent builds per precision matched **65,536 / 65,536 BF16
outputs** on Nicolas's RTL `266c593` pinned Spike model. Their bound MLIR,
physical commands, generated C, ELF, simulator logs, and full-output receipts
match between runs. See the [FP8 evidence](evidence/radiance_tilewise_vpu_affine_266c593/fp8/index.json),
[FP4 evidence](evidence/radiance_tilewise_vpu_affine_266c593/fp4/index.json),
and [regression test](../tests/test_tilewise_vpu_chain_evidence.py). The
[general source CLI receipt](evidence/radiance_tilewise_vpu_affine_266c593/fp8/cli_capture_index.json)
adds two capture-driven FP8 Spike builds with the same executable and result.
Only path-bearing link log hashes vary.

The FP8 matrix operands and starting BF16 golden come from the selected
Radiance C source, while the affine epilogue comes from the captured PyTorch
graph. The FP4 fixture is generated from pinned Radiance code. Neither
source driver contains this affine VPU sequence, so these runs establish
parity against a source-derived BF16 reference with VPU rounding, rather than
source ELF parity.
The current policy supports ordered in-place scalar operations on one
contraction's BF16 output tiles; it does not yet schedule arbitrary tensors,
broadcasts, reductions, or cross-engine dependencies. These results are
Spike evidence, not an FPGA result.

Reproduce the FP8 capture and execution with the pinned roots above:

```sh
python -m tools.qualify_radiance_tilewise_vpu_x2 \
  --model2mlir-root /path/to/model2MLIR-e9ded36 \
  --mxq-root /path/to/microscaling-quant-b4af543 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --epilogue affine --scalar-bits 0x4000 \
  --second-scalar-bits 0x3fc0 \
  --out-dir /tmp/radiance-tilewise-vpu-affine
```

Use `tools.qualify_radiance_fp4_derived_tilewise_vpu_x2` with the same options
for the generated FP4 fixture. The general `tools.qualify_source_mx` command
with `--tilewise-vpu-from-capture` shown above also recognizes the ordered
chain directly from these sidecars.

## Generated four-tile FP4 GEMM with tilewise VPU epilogue

The [FP4 qualifier](../tools/qualify_radiance_fp4_derived_tilewise_vpu_x2.py)
extends the same typed `matmul * 2.0` path to four 128×128 FP4 output tiles.
Radiance has no committed 256×256 FP4 driver. The qualifier copies the pinned
Radiance FP8 256×256 driver into an isolated fixture, changes its precision to
FP4 and K tile to 128, inserts the `A_in_hw` to `A_in` activation alias used
by Radiance's FP4 drivers, and runs Radiance's `gen_mxgemm_data.py fp4 256 256 256`
against a freshly built copy of its `mx_golden`. It checks the base driver,
generator, golden sources, derived driver, and generated header hashes. The
bundle uses `radiance_source_derived_gemm_fixture` and an explicit derivation
record; the binder, Python MLIR verifier, and bundle loader reject a missing
or disguised origin, while the native dialect accepts the declared origin.
This is compiler execution against a generated
fixture, not parity with a committed Radiance source ELF.

The corrected derived driver also builds as a Muon/Radiance `.soc.elf` with
the pinned Radiance source revision and its exact Gemmini software submodule.
The [build receipt](evidence/radiance_fp4_generated_tilewise_vpu_266c593/source_build/receipt.json)
archives that executable and build log; that receipt certifies the build only.
The source driver uses Muon scheduling, while the pinned MX Spike run executes
the compiler's Rocket/RoCC ELF. Its separate Cyclotron execution is reported
below.

The [two-run receipt](evidence/radiance_fp4_generated_tilewise_vpu_266c593/index.json)
records four compiler-issued VPU commands and **65,536 / 65,536 BF16 outputs**
matching the generated matrix golden after exact BF16 ×2 on pinned Spike.
The [bound MLIR](evidence/radiance_fp4_generated_tilewise_vpu_266c593/tilewise_bound.mlir),
[physical program](evidence/radiance_fp4_generated_tilewise_vpu_266c593/build/physical_program.json.gz),
and [Spike log](evidence/radiance_fp4_generated_tilewise_vpu_266c593/build/spike.log)
are archived. Source, frontend, command, object, ELF, extension, and numerical
hashes reproduce. The two raw artifact manifests differ only in the linker
warning's build path.

Reproduce from a checkout of Radiance `80f84ca`, model2MLIR `e9ded36`, and
Nicolas Gemmini `266c593`:

```sh
python -m tools.qualify_radiance_fp4_derived_tilewise_vpu_x2 \
  --model2mlir-root /path/to/model2MLIR-e9ded36 \
  --mxq-root /path/to/microscaling-quant-b4af543 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /tmp/radiance-fp4-generated-tilewise-vpu-x2
```

The same derived FP4 fixture now supports a captured finite BF16 scalar MULS
epilogue. With `--scalar-bits 0x3fc0`, two runs from compiler `c6b41c0`
matched **65,536 / 65,536 BF16 values** for `matmul * 1.5` on Nicolas's pinned
Spike. The [capture and receipts](evidence/radiance_fp4_generated_tilewise_vpu_scalar_266c593/index.json)
and [regression test](../tests/test_fp4_tilewise_vpu_scalar_evidence.py)
cover the generated fixture, original graph, bound MLIR, four physical VPU
commands, derived BF16 reference, standalone source, ELF, and Spike output.
Reproduce with the command above, adding `--scalar-bits 0x3fc0` and choosing a
fresh output directory. This case retains the generated-fixture scope; it does
not establish parity with a committed FP4 source driver.

## Direct row-major BF16 readout across output tiles

`mx_gemmini.readout_bf16` now accepts the typed
`memory_layout = "row_major_bf16"` selection. The physical lowerer writes each
tile's BF16 values directly into the logical row-major output buffer using
16-byte readout strips. The standalone checker compares that buffer to the
source golden in linear order; the caller no longer has to untile output.
The default tile-major layout remains available for existing objects. Invalid
layout names are rejected by both the native dialect verifier and the Python
profile verifier.

The [three-case evidence index](evidence/bf16_row_major_readout_266c593/index.json)
archives typed MLIR, source-bundle manifest, generated physical commands,
Rocket issuer, ELF, Spike log, two independent artifact receipts, and a
[regression audit](../tests/test_bf16_row_major_spike_evidence.py). The first
two cases also archive data-free RV64 RoCC objects with a row-major BF16 output
pointer ABI. All output bytes are written exactly once.

| Case | Output tiles | Source-derived outputs matched on pinned Spike | Scope |
|---|---:|---:|---|
| [FP4 generated 256×256×256, VPU ×2](evidence/bf16_row_major_readout_266c593/fp4_source_cli/artifact_manifest.json) | Four 128×128 | 65,536 BF16 | Derived Radiance fixture; no committed 256×256 FP4 source ELF parity |
| [FP8 256×256×256, VPU ×2](evidence/bf16_row_major_readout_266c593/fp8_tilewise_vpu/artifact_manifest.json) | Four 128×128 | 65,536 BF16 | Source `tk256` driver is excluded from Radiance's 128 KiB build |
| [FP8 128×128×512, retiled](evidence/bf16_row_major_readout_266c593/fp8_retile64/artifact_manifest.json) | Four 64×64 | 16,384 BF16 | Compiler retiling of source data; source driver uses a different tile schedule |

The FP4 case can be reproduced with the one-command source path after
restoring its archived generated header beside the
[derived driver](evidence/radiance_fp4_generated_tilewise_vpu_266c593/fixture/kernels/gemm_mxgemmini/mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.cpp):
copy the fixture directory to a writable location and decompress
`mxgemm.data.fp4.m256n256k256.h.gz` there as
`mxgemm.data.fp4.m256n256k256.h`.

```sh
python -m tools.qualify_source_mx \
  --mlir docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/profile_bound.mlir \
  --driver /path/to/restored/fixture/kernels/gemm_mxgemmini/mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.cpp \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --tilewise-vpu-x2 --bf16-output-layout row_major_bf16 \
  --out-dir /tmp/fp4-row-major
```

The two runs per case reproduce bound MLIR, commands, issuer, ELF, and Spike
log. Their linker warning records the output path, so only the `link.log`
hash differs. This is Rocket/Spike numerical evidence for the MX path; it
does not qualify Muon, the combined Radiance SoC, RTL timing, or FPGA behavior.

The [runtime object qualifier](../tools/qualify_runtime_fp4_row_major_object.py)
also links the archived data-free FP4 object to a separate caller. It invokes
the same `mx_issue` twice with different packed codes and E8M0 scales, then
compares both output matrices directly in row-major order. Its second payload
swaps complete M and N halves of the source-derived fixture. Both independent
builds match **131,072 / 131,072 BF16 values** on pinned Spike, including a
recheck that the first output survived the second invocation. The
[two-run receipt](evidence/fp4_row_major_runtime_object_266c593/index.json),
[generated caller](evidence/fp4_row_major_runtime_object_266c593/mx_runtime_driver.c),
[Spike log](evidence/fp4_row_major_runtime_object_266c593/spike.log), and
[audit](../tests/test_fp4_row_major_runtime_object_evidence.py) are archived.
From this repository, reproduce it with:

```sh
python -m tools.qualify_runtime_fp4_row_major_object \
  --object-dir docs/evidence/bf16_row_major_readout_266c593/fp4_source_cli \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp4-row-major-runtime
```

## Muon MMIO issuer handoff

Selected checked physical MX programs can now produce data-free Muon RV32
issuer objects. The [MMIO object emitter](../tools/emit_mx_mmio_object.py)
selects `rtl_alternating` scale scheduling for hardware, checks Radiance's
`mxgemmini_mmio.h` register offsets and instruction word, then maps each
physical completion fence to a CPU fence followed by a poll of the MX gateway
busy register. Gateway transactions use Muon's `sw.shared` and `lw.shared`
instructions. The emitter checks their disassembly counts against the physical
command stream. It also verifies `radiance.h` and applies its
`0x100000000` GPU-local to MX-visible global address bit to every buffer
operand, rejecting a pointer plus offset outside RV32 local address space.
The caller supplies every operand, scratch, output, and gateway
base address at runtime. It uses Muon's `rv32im_zfinx_zhinx` compiler with
the Vortex target feature, matching the source kernel build flags.

Two independent builds of the source-bound four-tile FP8+VPU program produced
identical [issuer C](evidence/fp8_vpu_muon_mmio_object_266c593/mx_issue.c.gz),
[RV32 object](evidence/fp8_vpu_muon_mmio_object_266c593/mx_issue.o),
[physical program](evidence/fp8_vpu_muon_mmio_object_266c593/physical_program.json.gz),
[disassembly](evidence/fp8_vpu_muon_mmio_object_266c593/disassembly.txt.gz),
and [manifest](evidence/fp8_vpu_muon_mmio_object_266c593/object_manifest.json).
The [audit](../tests/test_muon_mmio_object_evidence.py) checks all 21 fence and
busy-wait pairs, four VPU commands, the row-major output ABI, and the
archive hashes. The generated header also gives `mx_issue` C linkage when
included from a C++ Muon kernel. Reproduce the object with:

```sh
python -m tools.emit_mx_mmio_object \
  --mlir docs/evidence/bf16_row_major_readout_266c593/fp8_tilewise_vpu/bound.mlir \
  --bundle docs/evidence/radiance_tilewise_vpu_x2_266c593/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --radiance-root /path/to/radiance-kernels-80f84ca \
  --muon-clang /path/to/llvm-muon/bin/clang \
  --out-dir /tmp/fp8-vpu-muon-mmio
```

The generated FP4+VPU fixture produces a second deterministic
[issuer object](evidence/fp4_vpu_muon_mmio_object_266c593/mx_issue.o),
[physical program](evidence/fp4_vpu_muon_mmio_object_266c593/physical_program.json.gz),
and [manifest](evidence/fp4_vpu_muon_mmio_object_266c593/object_manifest.json).
Its tile-major BF16 output ABI follows the physical four-tile plan. The
[audit](../tests/test_fp4_muon_payload_evidence.py) checks its 1,134 commands,
33 completion fences, four VPU commands, and shared-gateway disassembly.
Reproduce with:

```sh
python -m tools.emit_mx_mmio_object \
  --mlir docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/tilewise_bound.mlir \
  --bundle docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --radiance-root /path/to/radiance-kernels-80f84ca \
  --muon-clang /path/to/llvm-muon/bin/clang \
  --out-dir /tmp/fp4-vpu-muon-mmio
```

The [Muon link probe](../tools/link_mx_muon_probe.py) compiles a C++ caller and
links the issuer with Radiance's Muon runtime and linker script. Two builds
produced an identical [RV32 ELF](evidence/fp8_vpu_muon_mmio_object_266c593/link_probe/link_probe.elf)
and [link receipt](evidence/fp8_vpu_muon_mmio_object_266c593/link_probe/link_manifest.json).
The probe guards its call with a zero-initialized volatile flag and has null
operands; it is for ABI and link validation only. Reproduce it after building
the MMIO object:

```sh
python -m tools.link_mx_muon_probe \
  --object-dir /tmp/fp8-vpu-muon-mmio \
  --radiance-root /path/to/radiance-kernels-80f84ca \
  --muon-clangxx /path/to/llvm-muon/bin/clang++ \
  --out-dir /tmp/fp8-vpu-muon-link-probe
```

No admitted Radiance MX+VPU SoC profile or RTL/FPGA execution receipt exists
for this path yet. The pinned Spike results above qualify the Rocket path
separately; a scoped Muon functional-model qualification follows below.

### Source-bound Muon kernel and simulator gap

The [payload linker](../tools/link_mx_muon_payload.py) binds a checked source
bundle to a generated Muon issuer. It embeds the four operand and scale arrays,
allocates BF16 output and scale scratch, and derives the VPU×2 BF16 reference
from the source golden in the selected output layout. Its kernel counts
mismatches across all 65,536 BF16 outputs after `mx_issue` completes. Two
independent FP8 builds produced identical
[RV32 ELFs](evidence/fp8_vpu_muon_payload_266c593/mx_kernel.elf) and
[manifests](evidence/fp8_vpu_muon_payload_266c593/payload_manifest.json).
Reproduce after emitting the MMIO object above:

```sh
python -m tools.link_mx_muon_payload \
  --object-dir /tmp/fp8-vpu-muon-mmio \
  --bundle docs/evidence/radiance_tilewise_vpu_x2_266c593/bundle \
  --radiance-root /path/to/radiance-kernels-80f84ca \
  --muon-clangxx /path/to/llvm-muon/bin/clang++ \
  --out-dir /tmp/fp8-vpu-muon-payload
```

Two independent FP4 builds likewise produced identical
[RV32 ELFs](evidence/fp4_vpu_muon_payload_266c593/mx_kernel.elf) and
[manifests](evidence/fp4_vpu_muon_payload_266c593/payload_manifest.json).
The FP4 verifier reorders the generated Radiance golden into the physical
tile-major output layout. Reproduce after emitting the FP4 issuer above:

```sh
python -m tools.link_mx_muon_payload \
  --object-dir /tmp/fp4-vpu-muon-mmio \
  --bundle docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/bundle \
  --radiance-root /path/to/radiance-kernels-80f84ca \
  --muon-clangxx /path/to/llvm-muon/bin/clang++ \
  --out-dir /tmp/fp4-vpu-muon-payload
```

The current Cyclotron model completes this ELF but produces zero output bytes:
the kernel reports 65,517 BF16 mismatches, confirmed independently from the
memory dump. The [diagnostic](evidence/fp8_vpu_muon_payload_266c593/cyclotron_diagnostic.json)
records the exact model and binary digests. Cyclotron's MX co-model implements
the source kernel's `loop_ws` path; it has no handlers for this compiler stream's
move-in (funct 2), move-out (funct 3), or VPU (funct 33), and treats scale DMA
(funct 27) as a no-op. The stock-model result is a simulator coverage gap.

An [isolated Cyclotron patch](evidence/fp8_vpu_muon_payload_266c593/cyclotron_compiler_stream.patch)
adds config strides, operand DMA, scale DMA including the FP4 alternate scale
bank, these fixtures' scalar-multiply VPU operation, and per-output-tile
accumulator clearing. The
[qualifier](../tools/qualify_mx_muon_cyclotron.py) clones pinned Cyclotron,
applies the patch, builds it, runs its 35 existing MX tests, then executes the
compiler-issued Muon ELF twice. Two independent fresh builds produced the same
simulator binary and memory dump. The [qualification receipt](evidence/fp8_vpu_muon_payload_266c593/patched_cyclotron_qualification.json)
and [archived memory dump](evidence/fp8_vpu_muon_payload_266c593/patched_cyclotron_gmem.bin.gz)
show **0 / 65,536 BF16 mismatches**, both from the kernel's verifier and an
independent byte-for-byte comparison with the derived source golden. Reproduce
this experimental result with:

```sh
python -m tools.qualify_mx_muon_cyclotron \
  --cyclotron-root /path/to/cyclotron-2d6adad \
  --payload-dir docs/evidence/fp8_vpu_muon_payload_266c593 \
  --bundle docs/evidence/radiance_tilewise_vpu_x2_266c593/bundle \
  --muon-llvm /path/to/llvm-muon \
  --out-dir /tmp/fp8-vpu-muon-cyclotron
```

The same qualifier independently executes the generated FP4 256×256×256
compiler Muon ELF. Two fresh patched-model builds each ran the ELF twice and
matched all **65,536 / 65,536 BF16 outputs**. The
[FP4 receipt](evidence/fp4_vpu_muon_payload_266c593/patched_cyclotron_qualification.json),
[reproduction receipt](evidence/fp4_vpu_muon_payload_266c593/patched_cyclotron_qualification_repro.json),
and [full memory dump](evidence/fp4_vpu_muon_payload_266c593/patched_cyclotron_gmem.bin.gz)
record both the kernel verifier and an independent comparison with the
source-derived, tile-major BF16 golden. Reproduce with:

```sh
python -m tools.qualify_mx_muon_cyclotron \
  --cyclotron-root /path/to/cyclotron-2d6adad \
  --payload-dir docs/evidence/fp4_vpu_muon_payload_266c593 \
  --bundle docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/bundle \
  --muon-llvm /path/to/llvm-muon \
  --out-dir /tmp/fp4-vpu-muon-cyclotron
```

These receipts qualify the selected FP8 and generated FP4 MX+VPU compiler
Muon MMIO paths on an explicitly patched functional model. Radiance has no
committed 256×256 FP4 source driver; the FP4 fixture comes from its pinned
generator. The model patch covers these command and VPU subsets. Other
precision modes, VPU operations, RTL cycles, and FPGA execution remain
separate gates.

### Executed generated FP4 source tiles on Cyclotron

The [source-tile qualifier](../tools/qualify_radiance_fp4_derived_cyclotron_tiles.py)
builds four Muon/Radiance 128×128 drivers from the pinned Radiance FP4
single-tile template. It slices the generated 256×256 fixture's packed A/B
bytes and E8M0 scales for each output tile; the BF16 golden stays outside
the executable. On the pinned
Cyclotron functional MX model with the separately qualified accumulator
overwrite correction, each tile matches **16,384 / 16,384 BF16 outputs**.
Assembled in matrix order, the four executed outputs match all **65,536**
generated BF16 values. Exact BF16 ×2 has the same SHA-256 as the compiler's
full-output Spike reference. Two independent source builds reproduce the
header, ELF, and output hashes. The [receipt](evidence/radiance_fp4_generated_cyclotron_tiles_266c593/receipt.json),
[source outputs](evidence/radiance_fp4_generated_cyclotron_tiles_266c593/assembled_bf16.bin.gz),
and [repeat receipt](evidence/radiance_fp4_generated_cyclotron_tiles_266c593/receipt_repro.json)
are archived.

The same qualifier also builds and runs the naive generated 256×256 source
driver. It has **65,490 / 65,536 BF16 mismatches** against the generated
matrix golden. Radiance's shared GEMM helper currently invokes one output
tile and leaves multi-tile operand offsets and scale loads unfinished; a
driver's 256×256 dimensions alone do not make it a four-tile source kernel.
The four separate source programs are an explicit derived decomposition.
Cyclotron is a functional model; this does not qualify an unmodified
multi-tile source driver, RTL, or FPGA execution of the VPU epilogue.

### Reusable four-tile FP4 MX+VPU object on Spike

The [object manifest](evidence/radiance_fp4_runtime_tilewise_object_266c593/object_manifest.json)
describes a linkable RV64 RoCC issuer generated from the typed four-tile FP4
MLIR. Its [object](evidence/radiance_fp4_runtime_tilewise_object_266c593/mx_issue.o)
has no embedded operand, scale, or golden data. The caller supplies A/B codes,
E8M0 scales, output, and scratch pointers. For four output tiles, the output
ABI is **output-tile-major BF16**; each 128×128 tile occupies a consecutive
32 KiB region. The object manifest now states that layout explicitly.

The [runtime qualifier](../tools/qualify_runtime_fp4_tilewise_object.py)
links this one object into a driver that calls it twice. The first call uses
the generated Radiance FP4 fixture. The second swaps its 128-row M halves and
128-column N halves in both packed operands and scales. Its independent
expected matrix is the corresponding permutation of the source BF16 golden,
followed by exact BF16 ×2. Both calls match every output on Nicolas's pinned
Spike extension: **131,072 / 131,072 BF16 values**, with the first output
still intact after the second call. The
[receipt](evidence/radiance_fp4_runtime_tilewise_object_266c593/index.json),
[repeat receipt](evidence/radiance_fp4_runtime_tilewise_object_266c593/index_repro.json),
[driver](evidence/radiance_fp4_runtime_tilewise_object_266c593/mx_runtime_driver.c),
[ELF](evidence/radiance_fp4_runtime_tilewise_object_266c593/mx_runtime_fp4_tilewise.elf.gz),
and [Spike log](evidence/radiance_fp4_runtime_tilewise_object_266c593/spike.log)
are archived. The repeat receipt matches byte for byte. This exercises runtime
rebinding and the multi-tile output ABI on the functional Spike model. The
256×256 FP4 fixture remains source-derived; Radiance has no committed
256×256 FP4 driver, and this run does not qualify Muon, RTL, or FPGA execution.

Reproduce with Nicolas Gemmini `266c593` and the selected RISC-V toolchain:

```sh
python -m tools.emit_mx_object \
  --mlir docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/tilewise_bound.mlir \
  --bundle docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp4-runtime-object
python -m tools.qualify_runtime_fp4_tilewise_object \
  --object-dir /tmp/fp4-runtime-object \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp4-runtime-run
```

### Reusable committed-source FP8 object on Spike

The [FP8 object](evidence/radiance_fp8_runtime_source_object_266c593/mx_issue.o)
comes from the committed Radiance 128×128×512 `tk256` FP8 source fixture's
bound typed MLIR. Its [manifest](evidence/radiance_fp8_runtime_source_object_266c593/object_manifest.json)
has no embedded operand or golden bytes and declares row-major BF16 output.
The [qualifier](../tools/qualify_runtime_fp8_source_object.py) calls the same
object with the original source codes and scales, then with valid M-row and
N-column half permutations. It permutes the independent source golden in the
same logical coordinates. Both runs match **32,768 / 32,768 BF16 outputs** on
Nicolas's pinned stock Spike extension; the first output remains intact after
the second call. The [receipt](evidence/radiance_fp8_runtime_source_object_266c593/index.json)
matches the [repeat](evidence/radiance_fp8_runtime_source_object_266c593/index_repro.json)
byte for byte. The [driver](evidence/radiance_fp8_runtime_source_object_266c593/mx_runtime_driver.c),
[ELF](evidence/radiance_fp8_runtime_source_object_266c593/mx_runtime_fp8_source.elf.gz),
and [Spike log](evidence/radiance_fp8_runtime_source_object_266c593/spike.log)
are archived. This tests Rocket object rebinding and full output parity for a
committed source case; Muon, RTL, and FPGA execution remain separate.

```sh
python -m tools.emit_mx_object \
  --mlir docs/evidence/radiance_fp8_512_tk256_latest_266c593/bound.mlir \
  --bundle docs/evidence/radiance_fp8_512_tk256_latest_266c593/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp8-runtime-object
python -m tools.qualify_runtime_fp8_source_object \
  --object-dir /tmp/fp8-runtime-object \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp8-runtime-run
```

### Reusable two-wave FP6 object with runtime LUTs on Spike

The [FP6 object](evidence/radiance_fp6_runtime_source_object_266c593/mx_issue.o)
comes from the source-bound generated Radiance 128×128×1024 FP6 fixture's
typed MLIR. Its [manifest](evidence/radiance_fp6_runtime_source_object_266c593/object_manifest.json)
declares pointers for packed A/B codes, E8M0 scales, **all three 64-line
A/B/C LUT banks**, BF16 output, and scratch. It embeds no operand or golden
bytes. The [qualifier](../tools/qualify_runtime_fp6_source_object.py) calls
the same object twice, first with the source fixture and then with a valid
M-row/N-column permutation of codes, scales, LUT banks, and the independent
BF16 golden. Both calls match **32,768 / 32,768 BF16 outputs** on Nicolas's
pinned stock Spike extension; the first output is still intact after the
second call. The [receipt](evidence/radiance_fp6_runtime_source_object_266c593/index.json)
matches the [repeat](evidence/radiance_fp6_runtime_source_object_266c593/index_repro.json)
byte for byte. The [driver](evidence/radiance_fp6_runtime_source_object_266c593/mx_runtime_driver.c),
[ELF](evidence/radiance_fp6_runtime_source_object_266c593/mx_runtime_fp6_source.elf.gz),
and [Spike log](evidence/radiance_fp6_runtime_source_object_266c593/spike.log)
are archived. A fresh object emission reproduced the issuer and
[manifest](evidence/radiance_fp6_runtime_source_object_266c593/object_manifest_repro.json)
byte for byte. This uses the compiler's `spike_serial` two-wave schedule because
the pinned Spike FP6 path ignores the alternating scale-bank selector. The
RTL-alternating path has its separate, isolated model-correction experiment
above. This run does not qualify Muon, RTL, or FPGA execution.

```sh
python -m tools.emit_mx_object \
  --mlir docs/evidence/radiance_fp6_fullout_266c593/fp6_128x128x1024/bound.mlir \
  --bundle docs/evidence/radiance_fp6_fullout_266c593/fp6_128x128x1024/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp6-runtime-object
python -m tools.qualify_runtime_fp6_source_object \
  --object-dir /tmp/fp6-runtime-object \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/fp6-runtime-run
```

Reproduce from a clean Radiance `80f84ca` checkout, its `6fc8ec7` MX software
submodule, the pinned Cyclotron overwrite-model worktree, and the Muon toolchain:

```sh
python -m tools.qualify_radiance_fp4_derived_cyclotron_tiles \
  --source-root /path/to/radiance-kernels-80f84ca \
  --radiance-lib-root /path/to/built-radiance/lib \
  --mx-software-root /path/to/radiance-kernels-80f84ca/lib/mxgemmini \
  --cyclotron-root /path/to/cyclotron-overwrite-probe \
  --llvm-muon /path/to/llvm-muon \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/radiance-fp4-derived-cyclotron-tiles
```

### DIM8/DIM32 source roster on Nicolas's Spike

`tools.qualify_radiance_mx_base_profile --all-fullout` selects all 23
BF16-output MX GEMM drivers in the pinned model2MLIR `e9ded36` capture. It
checks each driver and data header byte for byte, rebinds its typed handoff to
the selected Nicolas Rocket profile, lowers physical MX commands, builds an
RV64 ELF, and compares every output on the matching `gemmini_dim8` or
`gemmini_dim32` Spike extension. The eight quantized-output drivers are a
separate qualification scope.

The checked-in Radiance BF16 goldens use DIM16's accumulator schedule.
For DIM8/DIM32, the qualifier first recompiles Radiance's pinned host model
at DIM16 and requires its output to match the source golden. It then changes
the mesh geometry and accumulator schedule for the target. The full-roster
reference policy also incorporates Nicolas's `MxFPMul` product floor, which
flushes a product when its exponent is below −16. That rule is present in
Nicolas's Spike math header but absent from the older Radiance host header.
The target golden, source golden, both model digests, the transformed model
digests, and the product-floor policy are recorded in each bundle. A target
match is therefore a target-model result, not a claim that the original
DIM16 source golden bytes are identical.

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --all-fullout \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim8AllAsymGemminiRocketConfig.json \
  --out-dir /tmp/mx-dim8-fullout
```

Large physical streams use `-O0` only for the generated straight-line RoCC
issuer; the receipt records that choice. Smaller issuers and the runtime
support retain `-O2`.

The [archived matrix](evidence/radiance_mx_fullout_mesh_roster_266c593/index.json)
contains two independent Spike runs for every case. On each run, DIM8 and
DIM32 each matched all **23 / 23 drivers and 376,832 / 376,832 BF16 outputs**
across FP4, FP6, and FP8. The generated issuer, physical program, RV64 ELF,
payload provenance, and Spike log are saved per case; repeat receipts agree
after excluding build logs that embed their output directory. The DIM8
128×128×5632 FP8 case changes exactly one target-model value when Nicolas's
product floor is applied; the corrected target golden and the compiler ELF
then match all 16,384 values. DIM32 target goldens are unchanged by that
rule across this roster.

These two all-precision profiles do not include the VPU. A separate check
on Nicolas's `MxE4M3Fp4VpuGemminiRocketConfig` matched the selected FP8 and
FP4 source contractions twice, **20,480 / 20,480 BF16 outputs per run**.
The archive includes those two VPU-profile matrix cases; the vector command
tests above exercise the VPU instructions themselves. Nicolas's current
VPU profile supports FP8 and FP4, while FP6 uses a LUT-capable profile
without the VPU.

The companion `--all-requant` option selects the eight source drivers that
write quantized outputs. It derives target-mesh BF16 first, then applies
Radiance's checked FP8 output policy to FP8 and FP4 input kernels or its
per-row LUT projection to FP6 kernels. The bundle records both the original
source-header code/scale hashes and the derived target code/scale hashes.
The generated program reads BF16 from MX and executes the typed host
requantization epilogue before checking every output byte and E8M0 scale on
Spike. This path matches Radiance's output convention; the MX hardware
requantizer has a separate numerical contract.

The [requant archive](evidence/radiance_mx_requant_mesh_roster_266c593/index.json)
records two independent runs for DIM8, DIM16, and DIM32. Every profile
matched all **8 / 8 drivers, 90,112 / 90,112 output bytes, and 3,328 / 3,328
E8M0 scales per run**. Across the six runs, the compiler checked 540,672
output bytes and 19,968 scales. Normalized qualification rows, generated
ELFs, and Spike logs are identical across each pair of runs. The source and
target code/scale bytes, bound MLIR, physical program, and generated issuer
are archived per case. DIM16 compares the source headers directly; DIM8 and
DIM32 compare the derived target-mesh reference.

### Complete legal source roster on the MX+VPU profile

The `--precision` selector restricts either source roster to the formats
admitted by one RTL profile. For Nicolas's
`MxE4M3Fp4VpuGemminiRocketConfig`, FP8 and FP4 are legal; FP6 LUT compute
is rejected during handoff binding. The
[VPU-profile archive](evidence/radiance_mx_vpu_legal_roster_266c593/index.json)
records two direct Spike runs of every matching Radiance MX GEMM driver:
**18 / 18 BF16-output drivers and 294,912 / 294,912 BF16 values**, plus
**6 / 6 requant drivers, 73,728 / 73,728 output bytes, and 2,304 / 2,304
E8M0 scales** per run. Generated ELFs and Spike logs match between runs.
The archive also records the exact FP6 profile rejection.

These source drivers execute the matrix and optional host requantization
path under a VPU-capable profile; they contain no VPU instructions. The
separate VPU command and captured graph qualifications above exercise vector
execution. FP6+VPU remains a hardware configuration gap in Nicolas's pinned
branch.

### Direct Spike receipts for the remaining Rocket wrappers

The [eight-wrapper archive](evidence/nicolas_rocket_wrapper_matrix_266c593/index.json)
adds a direct named-profile run for each Chipyard Rocket wrapper previously
represented only by a mode-class probe. The selected DIM8 and DIM32 all-format
wrappers run E4M3-LUT × E2M3-LUT; the DIM32 base wrapper runs generated
FP4×FP4; the E2M3, E3M2, and E5M2 single-format wrappers run their matching
symmetric modes; and the two test wrappers run direct FP4×FP4. Latest pinned
model2MLIR `e9ded36` captures each contraction. The compiler binds the
source or Nicolas-generated packed data, emits physical commands and RV64
ELFs, then compares every output on Nicolas's stock Spike extension. Each of
the eight cases passes **4,096 / 4,096 BF16 outputs in two independent runs**,
or **65,536 checked outputs** in total. The archive keeps the captures,
bound MLIR, resource manifests, generated issuers, ELFs, logs, and both
receipts. A direct FP4 mode needs no LUT memory: the recipe validator now
accepts the no-LUT `TestMxGemminiRocketConfig` and the narrower LUT memory in
`TestRequantizerLutMxGemminiRocketConfig` for that mode. The latter test checks
its matrix path; its requantizer remains a separate qualification.
An independent [post-archive reproduction](evidence/nicolas_rocket_wrapper_matrix_266c593/reproduction_7809c82.json)
from compiler `7809c82` regenerated all eight profiles' bound MLIR, physical
programs, issuers, ELFs, and Spike logs with the same hashes as the archive.

On a fresh pinned Gemmini checkout with initialized software submodules,
materialize the two generated-header sets, then reproduce the eight runs:

```sh
python -m tools.generate_nicolas_missing_headers \
  --rtl-root "$MX_RTL_ROOT" --microxcaling-root "$MXQUANT_ROOT"
python -m tools.generate_nicolas_mesh_headers --mesh-dim 32 \
  --rtl-root "$MX_RTL_ROOT" --microxcaling-root "$MXQUANT_ROOT"
"$MODEL2MLIR_PYTHON" -m tools.qualify_mx_rocket_wrapper_matrix \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --jobs 3 \
  --out-dir /new/mx-rocket-wrapper-matrix \
  --baseline-index docs/evidence/nicolas_rocket_wrapper_matrix_266c593/index.json
```

The [profile catalog](evidence/mx_profile_qualification_catalog_266c593/index.json)
now indexes direct Spike evidence for all 40 Chipyard Rocket MX wrappers.
Its other 41 profiles are Gemmini fragments without a Chipyard wrapper. The
catalog keeps the three all-asymmetric 36-mode matrices as separate
mode-class probes and retains the stock Spike weight-LUT failure explicitly.
It also indexes the two named VPU Rocket profiles' connected 64³→64×32×64
full-output Spike receipts separately from the FP8/FP4 GEMM roster, which
contains no VPU commands. These simulator receipts do not change either
profile's structural qualification state or establish RTL/FPGA parity.

### The Nicolas requantizer wrapper's three output modes

The [requantizer-wrapper archive](evidence/nicolas_requantizer_wrapper_266c593/index.json)
exercises `TestRequantizerLutMxGemminiRocketConfig` beyond its BF16 matrix
receipt. Its FP8 and FP4 cases use the matching Radiance requant drivers,
captured through model2MLIR `e9ded36`, then bind their original handoff to
this exact profile. The compiler emits a quantized MX readout, physical RoCC
commands, and an RV64 ELF. Nicolas's stock Spike reports **4,096 FP8 codes
and 128 E8M0 scales**, then **2,048 packed FP4 bytes and 128 scales**, all
matching the profile-specific hardware oracle. The source headers use a
different output convention; the archive records those differences rather
than claiming byte parity with the Radiance header. The separate host
compatibility path does match that header but does not exercise the hardware
requantizer.

The FP6 case takes the checked-in 128×128×2048 **fullout** driver and
explicitly derives an E3M2 quantized terminal readout. It checks **8,192
packed index bytes and 512 scales** against Nicolas's Spike oracle. This is
direct evidence for the selected profile's FP6 LUT and requantized output
path, but it is not a captured FP6 Radiance requant source driver. The pinned
Spike's FP6 scale-selector issue still requires the serial scheduling mode.

Given a current `tools.recapture_radiance_roster` result, reproduce the three
cases with:

```sh
"$MODEL2MLIR_PYTHON" -m tools.qualify_nicolas_requantizer_wrapper \
  --capture-root /path/to/current-roster-capture \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --out-dir /new/mx-requantizer-wrapper \
  --baseline-index docs/evidence/nicolas_requantizer_wrapper_266c593/index.json
```

## Ordered scalar VPU operations between connected MX matrices

The connected 64³→64×32×64 FP8 lowerer now accepts 1–16 SSA-linked,
in-place BF16 scalar VPU operations before resident requantization and MM2.
Each operation is issued with a completion fence before the next operation
uses its scratchpad result. The public `tools.compile_object` command selects
this graph family and emits a data-free RV64 object.

The [two-operation Spike receipt](evidence/nicolas_connected_scalar_chain_266c593/index.json)
uses Nicolas's source-qualified MULS ×2 followed by an explicitly derived
ADDS +0. The independent BF16 reference is unchanged by the second operation.
On Nicolas's pinned Spike, the compiler-issued object matches all **4,096 C1
BF16 values, 4,096 C1 FP8 codes, 128 C1 scales, 2,048 C2 FP8 codes, and 64 C2
scales**. The BF16 readout is taken before the VPU operations; the quantized
C1 and C2 readouts are taken after them. This proves the second VPU command
and resident handoff execute in order for the identity case.
An independent checkout of the pushed `handwritten-implementation` branch
reproduced the bound MLIR, data-free object, linked ELF, and Spike log hashes;
the [published-clone receipt](evidence/nicolas_connected_scalar_chain_266c593/published_clone_replay.json)
records both compiler revisions and the matching artifact hashes.

The [nonzero ADDS receipt](evidence/nicolas_connected_scalar_chain_nonzero_266c593/index.json)
replays the same source-bound graph with a derived second VPU operation,
`ADDS +1.5` (`0x3fc0` BF16). The qualifier first checks that Nicolas's pinned
FP8 matrix model reproduces the unmodified C2 source golden. It then applies
the VPU BF16 reference, quantizes C1, and feeds those changed C1 codes and
scales plus the source B2 operands to the model for fresh C2 references.
Compared with the original source goldens, **4,082 C1 codes, 128 C1 scales,
2,015 C2 codes, and 64 C2 scales change**. A clean checkout of commit
`f492142` generated the data-free object and linked ELF; pinned Spike matched
every C1/C2 value against those derived references. This qualifies one
nonidentity ordered scalar chain on Spike. The extra ADDS is a derived
candidate, not an unchanged Nicolas source kernel.

Reproduce the object build, native dialect verification, link, and Spike run
from the checked-in Nicolas source capture with:

```sh
python -m tools.replay_connected_scalar_chain \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /tmp/mx-connected-scalar-chain
```

For the nonzero candidate, use a Python environment with PyTorch and add
`--adds-bf16 0x3fc0`. The replay refuses a changed Nicolas FP8 model and
records the source, model, profile, object, ELF, and Spike hashes.

## Remaining gates

1. Qualify the source-compatible FP8 and FP6 host epilogues on RTL or FPGA
   if those paths are needed there. The FP6 requant receipts use generated
   fixtures because the corresponding headers are absent upstream; check
   future committed headers against them. Qualify the quantized-output
   epilogues on RTL or FPGA where required, remaining mode and memory
   configuration combinations without direct receipts, and extend multi-output
   vector tiling beyond the qualified FP8 and generated FP4 BF16 ×2, scalar
   ADDS, and ordered affine epilogues.
   For GQA, reconcile the
   source generator with the hardware product and accumulator precision,
   then requalify unchanged source bytes before claiming attention parity.
2. Generalize the connected chain's explicit scratchpad lifetimes beyond the
   qualified 64³ and 64×32×64 MX+VPU cases and the plain MX 16-row prefix ladder of
   Nicolas's 128³ source and the source-derived 16×96×96 and 64×96×96 plain
   MX cases.
   Two rectangular 64×96×64 → 64×{32,64}×96 pairs are now qualified on
   Spike, including distinct B1/B2 footprints; lower broader rectangular
   and mixed-engine graphs without a source-specific seam.
3. Check the candidate Spike weight-LUT lane fix against RTL, then qualify
   the one failing E4M3-direct × E4M3-LUT cell on DIM8, DIM16, and DIM32.
   The other 35 / 36 legal cells pass stock Spike on all three geometries;
   further RTL qualification remains separate. Keep unsupported profile
   combinations rejected. FP6+VPU requires a new RTL configuration and
   profile before it can be advertised.
4. Validate the experimentally matched alternating FP6 scale path against
   RTL, then qualify the Radiance MMIO/FPGA issue path separately from Rocket
   RoCC.

## Nicolas two-tile MX+VPU scheduling source baseline

The public [`tools.compile_object`](../tools/compile_object.py) command now
selects the full three-site graph as `full_vpu_branch`. It reuses the existing
two-branch physical lowerer and emits a data-free RV64 RoCC object. The input
is the complete typed graph plus its checked BF16-preload precursor, seven
source-derived `.bin` inputs (six runtime operands and a C1 BF16 reference),
and the [explicit buffer ABI](../examples/full-vpu-branch-abi.json). The
precursor establishes that the complete graph was formed by replacing its
preload with the captured first matrix; it is not executed by the object.
The compiler rejects a changed precursor, payload digest, profile, schedule,
or buffer map before emitting a qualified object.

The replay command below derives the operands from Nicolas's pinned source,
compiles serial and pipelined graphs on both VPU Rocket profiles through the
public entry point, checks each issuer against the previously qualified object,
and runs every object against the full source goldens on pinned Spike:

```sh
python -m tools.replay_full_vpu_branch_dispatch \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/mx-public-branch-replay
```

This integration covers Nicolas's 64×64 three-site, two-branch FP8 graph and
the two named VPU profiles. Broader graph shapes and branch structures still
need the lowerer's scratchpad lifetime rules generalized.
The [published-checkout replay](evidence/mx_full_vpu_branch_public_3619043/index.json)
records four public objects, their exact matches to the earlier source-bound
issuers, and fresh pinned Spike runs. Across the four runs, the source
comparisons cover **16,384 first-matrix BF16 values, 65,536 FP8 codes, and
2,048 E8M0 scales**, with no mismatches. The archive includes the actual
input and golden bytes, objects, physical streams, ELFs, and simulator logs.

Nicolas's `chain_pipelined.c` starts from a preloaded 64×64 BF16 C1 tile,
shares one B2 weight tile, and issues two resident VPU→requant→MM2 chains
with BF16 scalar factors 2 and 4. Its fenced, program-order, and pipelined
schedules test the reservation-station issue order. The
[pinned Spike source receipt](evidence/nicolas_chain_pipelined_266c593/source_spike_receipt.json)
records four successful checks (including warmup), with no clamped scales
skipped. Spike reported 635, 609, and 405 cycles for those three schedules;
these are model measurements and do not establish RTL overlap.

The [model2MLIR capture](evidence/nicolas_chain_pipelined_266c593/receipt.json)
uses pinned upstream `e9ded36` and has three FP8 contraction sites with no
opaque calls. Its [original graph](evidence/nicolas_chain_pipelined_266c593/original_graph.json)
shows one MM1 feeding distinct ×2 and ×4 branches, with B2 shared by both
MM2 sites. The capture's profile-bound MLIR is a three-site handoff; it does
not preserve branch SSA edges. `audit_chain_pipelined` checks Nicolas's exact
tile placement and issue-order markers and independently derives full C1 and
C2 FP8 codes and E8M0 scales for both factors from the source BF16 goldens.
The compiler-issued source chain is described below.

### Compiler-issued two-tile source chain

The [typed connected graph](evidence/nicolas_chain_pipelined_compiled_266c593/connected.mlir)
binds Nicolas's BF16 C1 preload to two VPU scalar operations, two resident
requants, and two MM2 sites. The physical lowerer reuses the qualified VPU,
SPAD_REQUANT, and resident-MM2 command builders and loads B2 only once. It
issues the branches in program order with dependency fences. The
[linkable object](evidence/nicolas_chain_pipelined_compiled_266c593/mx_issue.o)
contains no allocated data; its [physical program](evidence/nicolas_chain_pipelined_compiled_266c593/physical_program.json)
and [manifest](evidence/nicolas_chain_pipelined_compiled_266c593/object_manifest.json)
record the profile, source inputs, command stream, and object hashes.

The [standalone Spike log](evidence/nicolas_chain_pipelined_compiled_266c593/spike.log)
reports **0 mismatches** across both tiles' **16,384 FP8 codes** and **512
E8M0 scales**, checked against references derived from Nicolas's header. Two
independent compiler runs produced identical bound MLIR, physical program,
issuer, object, ELF, extension, and Spike log hashes. The executable starts
from the source driver's preloaded C1 BF16 tile; the captured upstream MM1 is
not issued by this object. The source's pipelined issue order and RTL cycle
overlap remain separate gates.
An independent checkout of pushed commit `8d16354` reproduced those hashes,
including the compiler source closure, in the
[fresh replay manifest](evidence/nicolas_chain_pipelined_compiled_266c593/fresh_replay_manifest.json).

Reproduce the object and Spike check with `python -m
tools.compile_nicolas_chain_pipelined --capture-dir
docs/evidence/nicolas_chain_pipelined_266c593 --rtl-root <Nicolas RTL>
--riscv-root <RISC-V tools> --mx-opt build/tools/mx-gemmini-opt --out-dir
<new output directory> --run-spike` from this repository.

### Compiler-issued complete shared-MM1 chain

`--include-mm1` extends the same compile command to the
[complete three-site typed graph](evidence/nicolas_chain_pipelined_full_266c593/connected.mlir).
The compiler issues MM1 from Nicolas's checked packed A1/B1 codes and scales,
reads its BF16 C1 tile, keeps tile 0 resident, and reloads that computed BF16
tile for the ×4 branch. Both branches share one B2 transfer. The source
driver's BF16 C1 preload is absent from the issuer command stream.

The [full object manifest](evidence/nicolas_chain_pipelined_full_266c593/object_manifest.json)
records zero allocated data bytes in the linkable issuer object and the six
runtime input hashes. The [pinned Spike log](evidence/nicolas_chain_pipelined_full_266c593/spike.log)
reports **0 mismatches** for all **4,096 MM1 BF16 values**, **16,384 C1/C2
FP8 codes**, and **512 E8M0 scales** across the two branches. This closes the
numerical shared-MM1→VPU→requant→MM2 path for Nicolas's 64³ source data. The
default compiler schedule uses dependency fences and program order. The
compiler's optional pipelined schedule has a separate functional Spike receipt
below; its RTL timing remains unqualified.
The [fresh checkout replay](evidence/nicolas_chain_pipelined_full_266c593/fresh_replay_manifest.json)
from pushed commit `495fefe` matched the bound graph, physical commands,
issuer, object, ELF, extension, Spike log, and compiler source closure hashes.

Append `--include-mm1` to the compile command above to reproduce the complete
chain. The output includes the bound graph, physical commands, data-free
object, standalone ELF, and full-output Spike check.

The separate
[E4M3-only VPU profile archive](evidence/nicolas_chain_pipelined_e4m3_only_266c593/compiled/object_manifest.json)
repeats the latest-model2MLIR capture, handwritten source oracle, and full
three-site compiler run under `MxE4M3VpuGemminiRocketConfig`. Its compiler
program also matches **4,096 C1 BF16 values**, **16,384 FP8 codes**, and
**512 scales** on Nicolas's pinned Spike. A second build reproduced its
bound graph, commands, object, ELF, and log hashes. The E4M3-only and
FP4-capable VPU profiles bind different profile hashes, while this FP8
program's issuer, object, ELF, and log are byte-identical. The
[profile qualification catalog](evidence/mx_profile_qualification_catalog_266c593/index.json)
indexes these as separate named-profile receipts; it does not infer FP4 or
FP6 VPU support from the shared FP8 result.

### Compiler-issued pipelined issue order

`--include-mm1 --issue-schedule pipelined` emits the source's stage order:
tile 1's BF16 reload precedes tile 0's VPU; tile 1's VPU is issued before
tile 0's MM2; tile 1's requant and MM2 follow. There are no completion
fences between those six VPU, requant, and MM2 stage commands. The
[FP4-capable VPU receipt](evidence/nicolas_chain_pipelined_full_266c593/pipelined/object_manifest.json)
and [E4M3-only VPU receipt](evidence/nicolas_chain_pipelined_e4m3_only_266c593/compiled_pipelined/object_manifest.json)
each match all **4,096 BF16 values**, **16,384 FP8 codes**, and **512 scales**
on Nicolas's pinned Spike. The profile catalog indexes the pipelined runs
separately from the fenced program-order runs. These are functional Spike
results; RTL queue overlap and the source's 405-cycle measurement have not
been reproduced by a compiler-issued RTL run.
Fresh checkouts of pushed commit `4262ff3` reproduced the issuer, object,
ELF, extension, Spike log, and compiler source closure hashes for both the
[FP4-capable](evidence/nicolas_chain_pipelined_full_266c593/pipelined/fresh_replay_manifest.json)
and [E4M3-only](evidence/nicolas_chain_pipelined_e4m3_only_266c593/compiled_pipelined/fresh_replay_manifest.json)
VPU profiles.

### Source-preloaded pipelined issue on Nicolas's Spike

The source driver starts from a BF16 C1 tile already in host memory. Running
the compiler with `--issue-schedule pipelined` and without `--include-mm1`
keeps that entry point: it loads B2 once, issues both BF16 tile transfers,
then issues VPU0, requant0, VPU1, MM2_0, requant1, and MM2_1. There are no
fences between the BF16 transfers or these six stage commands. The object is
linkable and contains no embedded operands or goldens.

The [FP4-capable VPU receipt](evidence/nicolas_chain_pipelined_compiled_266c593/preloaded_pipelined/object_manifest.json)
and [E4M3-only VPU receipt](evidence/nicolas_chain_pipelined_e4m3_only_266c593/compiled_preloaded_pipelined/object_manifest.json)
each compare **16,384 FP8 codes and 512 E8M0 scales** with zero mismatches on
Nicolas's pinned Spike. Their profile hashes differ; for this FP8 path their
issuer objects, ELFs, and Spike logs have identical hashes. Fresh checkouts of
compiler commit `dd39055` reproduced both archives' MLIR, physical programs,
objects, ELFs, and logs. The profile catalog records these as source-preloaded
pipeline receipts, distinct from the compiler-issued MM1 receipts. These
functional runs do not establish RTL queue overlap or a comparable cycle
count with Nicolas's handwritten source driver.

To reproduce either profile, use the compile command above with
`--issue-schedule pipelined --run-spike` and omit `--include-mm1`. For the
E4M3-only profile, also select its archived capture and profile JSON as in
the full-chain command.

### Source FP4 scratchpad requantization on Nicolas's Spike

Nicolas's `spad_requant_fp4.c` generates a 64×128 BF16 tile and checks FP4
codes against its own E2M1 reference in both flat and operand-A tiled layouts.
The compiler now lowers one typed `mx_gemmini.spad_requant` operation for each
layout from the same SSA input. Its data-free RV64 object issues the BF16
transfer, both FP4 requant commands, and both packed-image readouts. The
qualification driver retains Nicolas's input generator and every code and
scale comparison; a recorded [source patch](evidence/nicolas_spad_requant_fp4_266c593/fp4_capable/compiler_driver.patch)
replaces the handwritten accelerator issue sites with a call to that object.
The typed module passes native `mx-gemmini-opt` verification. This is a
target-operation diagnostic; the Radiance workload captures remain the
model2MLIR source-parity suite.

The [FP4-capable VPU receipt](evidence/nicolas_spad_requant_fp4_266c593/fp4_capable/receipt.json)
and [E4M3-only VPU receipt](evidence/nicolas_spad_requant_fp4_266c593/e4m3_only/receipt.json)
each record zero mismatches for **16,384 FP4 codes and 512 E8M0 scales** in
both the original source ELF and compiler-issued ELF on Nicolas's pinned
Spike. The compiler driver poisons all output buffers before issue, so the
checks also catch incomplete stores. Both compiler objects contain zero
allocated data bytes. Fresh checkouts of compiler commit `3117049` reproduced
each complete receipt and
all ten retained MLIR, command, object, driver, ELF, and log artifacts; see
the archived [FP4-capable replay](evidence/nicolas_spad_requant_fp4_266c593/fp4_capable/fresh_replay.json)
and [E4M3-only replay](evidence/nicolas_spad_requant_fp4_266c593/e4m3_only/fresh_replay.json).
The profile catalog indexes these two runs separately. The E4M3-only profile
has the FP4 requant instruction even though it does not have FP4 matrix
compute. The source and compiler cycle measurements cover different regions,
so they are not a performance comparison or RTL timing evidence.

```sh
python -m tools.qualify_nicolas_spad_requant_fp4 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/mx-fp4-dual-requant
```

Add `--profile profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json`
for the E4M3-only build. Pass `--baseline-receipt` with the matching archived
receipt to require identical digests from an independent checkout.
