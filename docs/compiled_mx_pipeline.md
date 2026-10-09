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
| FP8 128×128×256, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp8_128x128x256_20261009.json) |
| FP8 256×256×256, four 128×128 output tiles | MX+VPU E4M3/FP4 | 65,536 | [receipt](evidence/compiled_mx_fp8_256x256x256_20261009.json) |
| FP4 64×64×128, K tile 64 | MX+VPU E4M3/FP4 | 4,096 | [receipt](evidence/compiled_mx_fp4_64x64x128_20261009.json) |
| FP4 128×128×128, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp4_128x128x128_20261009.json) |
| FP4 128×128×256, K tile 128 | MX+VPU E4M3/FP4 | 16,384 | [receipt](evidence/compiled_mx_fp4_128x128x256_20261009.json) |
| FP6 128×128×2048, K tile 128 | E3M2 LUT, no VPU | 16,384 | [receipt](evidence/compiled_mx_fp6_128x128x2048_20261009.json) |

Each receipt records a zero-mismatch pinned Spike run. Independent output
directories reproduced identical payload-bound MLIR, physical source files,
objects, ELF, extension, and Spike log hashes for FP8 128×128×512 with both
K tile sizes and FP4 64×64×128.
The FP8 K tile 256 run tests a distinct schedule over the same source data;
the 256-deep runs use fresh PyTorch/model2MLIR captures and generated source
headers.
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

Nicolas's reference `vpu_ops`, `vpu_softmax`, and
`chain_vpu_spad_requant` programs were also built and run directly against
the pinned Spike extension. They passed all reference comparisons, including
the fused VPU cases and the VPU→requant→matmul chain. That is model capability
evidence; the [reference receipt](evidence/nicolas_vpu_spike_reference_20261009.json)
records source, ELF, tool, and log hashes. The dialect's physical
VPU/SPAD_REQUANT command lowerer is checked separately. The x2 epilogues above
establish one compiler-generated matrix/VPU composition; the full
VPU→requant→matmul chain still needs compiler output and numerical parity.

## Remaining gates

1. Reconcile the FP8/FP4 source requant goldens with Nicolas's current convention,
   then qualify FP6 quantized output, asymmetric legal modes, and the
   remaining source shapes. Extend multi-output tiling beyond the qualified
   FP8 BF16 shape.
2. Compose the full matrix→VPU→SPAD_REQUANT→matrix chain in one MLIR program
   with explicit scratchpad lifetimes and source numerical goldens.
3. Qualify every legal mode class on the matching Spike/RTL configuration,
   and keep unsupported profile combinations rejected. FP6+VPU requires a
   new RTL configuration and profile before it can be advertised.
4. Validate the alternating FP6 scale path against RTL, then qualify the
   Radiance MMIO/FPGA issue path separately from Rocket RoCC.
