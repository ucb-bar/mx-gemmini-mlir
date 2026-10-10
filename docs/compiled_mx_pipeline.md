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

A third run used that command at compiler `66df445`. Its
[reproduction receipt](evidence/radiance_mx_gemm_latest_e9ded36_ee22/one_command_66df445/reproduction.json)
and per-driver receipts show the same captured MLIR, physical source files,
objects, ELFs, extensions, and Spike logs as the archived baseline for all
31 drivers. The command refuses a changed source/tool revision or artifact
digest before writing a success receipt.

Repeat the capture and Spike commands with fresh output directories, then run
`tools.archive_radiance_roster` with both pairs and the header materialization
report. It rejects changed frontend artifacts, generated sources, objects,
ELFs, extensions, or Spike logs before making a reviewable archive.

This is full source parity for the current MX **GEMM driver roster** on pinned
Spike. It does not cover the mixed MX+Muon HBM FlashAttention kernel, arbitrary
typed graphs, FPGA execution, or the stock Spike E4M3-direct × E4M3-LUT defect
described above.

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

## Remaining gates

1. Qualify the source-compatible FP8 and FP6 host epilogues on RTL or FPGA
   if those paths are needed there. The FP6 requant receipts use generated
   fixtures because the corresponding headers are absent upstream; check
   future committed headers against them. Qualify remaining configuration
   families and source shapes without receipts, and extend multi-output
   tiling beyond the qualified FP8 BF16 shape. For GQA, reconcile the
   source generator with the hardware product and accumulator precision,
   then requalify unchanged source bytes before claiming attention parity.
2. Generalize the connected chain's explicit scratchpad lifetimes beyond the
   qualified 64³ Nicolas source case, and lower other typed graphs without a
   source-specific seam.
3. Check the candidate Spike weight-LUT lane fix against RTL, then qualify
   the one failing E4M3-direct × E4M3-LUT cell on DIM8, DIM16, and DIM32.
   The other 35 / 36 legal cells pass stock Spike on all three geometries;
   further RTL qualification remains separate. Keep unsupported profile
   combinations rejected. FP6+VPU requires a new RTL configuration and
   profile before it can be advertised.
4. Validate the alternating FP6 scale path against RTL, then qualify the
   Radiance MMIO/FPGA issue path separately from Rocket RoCC.
