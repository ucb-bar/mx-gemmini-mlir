# Nicolas MX source-program coverage

Nicolas's pinned `gemmini-mx-cleanup` revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52` lists **171 MX test
programs** in `software/gemmini-rocc-tests/bareMetalC/Makefile` (SHA-256
`fbf15aa6938b7da6b701a628292123ec652c31e7841f692005871e9a13c3c76d`).
All 171 listed names resolve to C files in that checkout. The roster has 74
`matmul_tiled_asym_*` programs, 88 other `matmul_tiled_*` programs, two
`chain_*`, two `vpu_*`, two `spad_requant*`, and three other MX programs.
The separate default Gemmini tests include convolution and other operations;
they are not in this MX roster.

The [machine-readable source inventory](evidence/nicolas_mx_source_inventory_266c593/index.json)
pins every listed C file's SHA-256 and searches archived receipts for that
exact hash. **All 171 programs have at least one direct source-hash reference.**
The last reference is a [source-only diagnostic](evidence/nicolas_single_tile_source_audit_266c593/README.md),
not a compiler execution receipt. A matching hash establishes provenance only. Some referenced
receipts are source-oracle runs or frontend captures, so neither group should
be read as a count of compiler-regenerated programs. Rebuild the inventory
with `python -m tools.audit_nicolas_mx_roster --rtl-root "$MX_RTL_ROOT"
--out docs/evidence/nicolas_mx_source_inventory_266c593/index.json --check`.
The [program-level regeneration audit](evidence/nicolas_mx_regeneration_audit_266c593/README.md)
validates selected-path generated-object Spike receipts for 156 matrix/LUT
names, generated-executable receipts for all six connected FP4/FP6/FP8 matrix
chains, and complete selected outputs for VPU softmax, both scratchpad
requant programs, both connected VPU chains, the memory readout, the full
VPU operation stream, and the Rocket numerical replay of the legacy FP6
MMIO debug data. The single-tile debug program remains separate. Its categories
retain each receipt's actual scope rather than treating every source-hash match as a full
program qualification.

The [15-case direct suite](evidence/nicolas_direct_matrix_suite_9df3384_266c593/README.md),
[three FP8 cases](evidence/nicolas_direct_fp8_three_789d192_266c593/README.md),
and [16 DIM8/16/32 cases](evidence/nicolas_direct_sixteen_28936e2_266c593/README.md)
cover **34 distinct direct matrix programs** through fresh model2MLIR
capture, public object compilation, and full-output Spike checks. Their
indices record 300,896 comparisons across BF16 values, quantized bytes, and scales.
The [alternate 32³ FP8 transfer replay](evidence/nicolas_fp8_alternate_mvin_compiled_5a2dd35_266c593/README.md)
adds one distinct source program and 1,024 BF16 comparisons. Its audited
transfer stream and generated object are identical to the existing 32³ case.
The [two FP8 128³ I-chunk replays](evidence/nicolas_fp8_chunked_public_fbe26eb_266c593/README.md)
add two named programs and 32,768 BF16 comparisons. Their source executables
and generated objects pass separately; the issued scale and loop commands are
also audited against the pinned source formulas.
The [direct 2D-scale I-chunk replay](evidence/nicolas_fp8_chunked_2d_public_287593c_266c593/README.md)
adds one named program and 16,384 BF16 comparisons. It emits the same object
as the two-chunk replay after a checked command-equivalence audit.
The [native DRAM-loop replay](evidence/nicolas_fp8_native_dram_public_1ef6f85_266c593/README.md)
adds one named program and 16,384 BF16 comparisons with LoopMatmul's own
operand loads and output store. Its original source and generated object pass
independently on pinned Spike.
The [two native N-chunk DRAM-loop replays](evidence/nicolas_fp8_native_nc_public_57b26ce_266c593/README.md)
add two named programs and 32,768 BF16 comparisons with one resident A load,
alternating B banks, and direct C slice stores.
The [two loop-managed-scale replays](evidence/nicolas_fp8_native_ls_public_126f138_266c593/README.md)
add two named programs and 32,768 BF16 comparisons with per-loop A/B scale
pointers and no standalone scale upload.
The [native K-tiled replay](evidence/nicolas_fp8_native_kt_public_af0b4ad_266c593/README.md)
adds one named program and 16,384 BF16 comparisons with two accumulating K
loops per N chunk.
The [two native scale-control replays](evidence/nicolas_fp8_native_scale_control_public_c2e25ce_266c593/README.md)
add two named programs and 32,768 BF16 comparisons. One uses direct 2D scale
uploads; the other uses the hardware scale-wait bit without an upload fence.
The [native store-ReLU replay](evidence/nicolas_fp8_native_relu_public_c729276_266c593/README.md)
adds one named program and 16,384 BF16 comparisons against the source's
ReLU-adjusted golden.
The [FP6 128³ repeated-LUT replay](evidence/nicolas_plain_fp6_128_aliased_lut_public_6c6b27a_266c593/README.md)
adds one named program and 16,384 BF16 comparisons while preserving the
source's original packed indices and all 64 A/B/C LUT lines.
The [FP8 64³ zero-base scratchpad replay](evidence/nicolas_fp8_smem_zero_public_64ecacd_266c593/README.md)
adds one named program and 4,096 BF16 comparisons. The source and compiler
store C at scratchpad row zero; their readout transports differ.
The [three DRAM-mvout Spike fallback replays](evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593/README.md)
add three named programs and 24,576 BF16 comparisons. Each source's `SPIKE_SIM`
path stores C at scratchpad row zero, and the generated object matches the
entire golden. The hardware accumulator-to-DRAM branches remain unqualified.
The [three accumulator command objects](evidence/nicolas_accumulator_readout_objects_266c593/README.md)
now select an accumulator source on the typed BF16 readout and reproduce each
source geometry's compute destination and MVOUT addresses in data-free RV64
Rocket objects. They are command-generation evidence only: Nicolas's hardware
C path uses an MMIO gateway and constant `0x7f` scale SRAM values, whereas
these objects use RoCC and source-header scales. Stock Spike cannot execute
the accumulator destination, so hardware numerical and timing parity are
still open.
An [experimental accumulator Spike extension](evidence/nicolas_accumulator_candidate_spike_266c593/README.md)
executes those three generated objects and matches all 24,576 BF16 source
goldens under a candidate MX shadow-result readout model. This checks the
compiler's addresses and complete results under that model; it does not
qualify RTL or FPGA packing, MMIO scale behavior, or timing.
A [controlled constant-scale replay](evidence/nicolas_accumulator_constant_scale_divergence_266c593/README.md)
passes the hardware branch's `0x7f` scale values to the same generated objects.
Under the same candidate Spike model, 24,516 of 24,576 outputs then differ
from the source-header goldens. The header-golden pass therefore cannot be
used to infer numerical parity for Nicolas's constant-scale hardware branch.
The [three same-format LUT source programs](evidence/nicolas_symmetric_lut_public_4e30dcf_266c593/README.md)
also pass through public data-free objects with their dedicated E2M3, E4M3,
and E5M2 Rocket profiles, matching 12,288 BF16 values. Three more
[E4M3 LUT shape and mesh replays](evidence/nicolas_e4m3_lut_shapes_public_4f316ac_266c593/README.md)
match 24,576 BF16 values on DIM8 and DIM32. Together, these exhaustive
suites cover the numerical results of **129 distinct named MX
source programs** from the 171-program roster.
The [E4M3 LUT requantized public-object replay](evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/README.md)
adds one named program with all 2,048 packed LUT-index output bytes and 128
E8M0 scales checked on Spike. [Four DIM8/DIM32 shape replays](evidence/nicolas_e4m3_lut_requant_mesh_public_4553528_266c593/README.md)
add 20,480 packed bytes and 1,280 scales across four more named programs.
[Two rectangular DIM32 replays](evidence/nicolas_e4m3_lut_requant_rect_public_75524c6_266c593/README.md)
add 8,192 packed bytes and 512 scales. The [E5M2 DIM16 packed LUT replay](evidence/nicolas_e5m2_lut_requant_public_8e9706f_266c593/README.md)
and [E2M3 DIM16 packed LUT replay](evidence/nicolas_e2m3_lut_requant_public_9df3a5e_266c593/README.md)
each match another 2,048 packed bytes and 128 scales.
[Ten E2M3/E5M2 mesh replays](evidence/nicolas_e2m3_e5m2_lut_requant_mesh_public_b94c3af_266c593/README.md)
add 40,960 packed bytes and 2,560 scales.
[Five E3M2 DIM8/DIM32 mesh replays](evidence/nicolas_e3m2_lut_requant_public_5b6dfc0_266c593/README.md)
add 20,480 packed bytes and 1,280 scales. These matrix/LUT numerical replay
suites cover **156 of 171 named MX source programs** for their selected Spike
execution paths. The six connected chains and six specialized programs bring
the audited selected-output count to **168 of 171**. The full
[`vpu_ops` program replay](evidence/nicolas_vpu_ops_program_public_266c593/README.md)
raises that count to **169 of 171**: one compiler-issued object runs all 30
ordered VPU commands and matches 13,056 BF16 values against the source reference
under both VPU Rocket profiles. The [legacy generic FP6 data replay](evidence/nicolas_ws_generic_portable_a042643_266c593/README.md)
raises the count to **170 of 171**: clean current model2MLIR emits the portable
matmul, and the generated Rocket object matches 16,384 BF16 values, 8,192
packed bytes, and 512 scales from the debug header. Its original fixed MMIO
issue path remains unqualified. The [single-tile debug source audit](evidence/nicolas_single_tile_source_audit_266c593/README.md)
pins the remaining C source and explains why its printed PASS is not a
numerical oracle: the active code loads only tile `(1,1)`, uses constant scales,
and has the BF16 comparison loop commented out. It has no compiler result receipt.
The [earlier 11-case suite](evidence/nicolas_direct_matrix_suite_9f3a759_266c593/README.md)
remains archived. The 171 source-hash-covered programs above include many weaker
forms of evidence and are not a count of compiler-regenerated programs.

The [64³ resident FP8 chain replay](evidence/nicolas_plain_fp8_chain64_public_52e6132_266c593/README.md)
adds `matmul_tiled_fp8_64x64_chain` to the separate connected-program
qualifications. A two-matmul PyTorch graph captured with pinned model2MLIR
lowers to a data-free RV64 MX object. On Nicolas's pinned Spike, the original
C source and the compiler-linked program each pass; the latter matches all
4,096 C1 and 4,096 C2 FP8 codes plus 128 E8M0 scales at each stage. This is
one of the six connected programs beyond the 156 matrix/LUT replay count.

| Source family | Compiler evidence | Coverage boundary |
|---|---|---|
| Asymmetric matrix modes | [74-program public-object suite](evidence/nicolas_asym_public_suite_e6923e8_266c593/README.md), [mode matrices](compiled_mx_pipeline.md#dedicated-dim16-asymmetric-mode-matrix), and [stock-model gap](compiled_mx_pipeline.md#isolated-weight-lut-spike-correction-across-all-legal-modes) | All 74 pinned `matmul_tiled_asym_*` C programs match 360,960 BF16 source outputs through generated public objects on Spike. This is named-program numerical coverage; separate legal-mode tests still expose the stock Spike gap, and no FPGA timing parity is claimed. |
| Same-format LUT E2M3/E3M2/E4M3/E5M2 | [Three BF16 mode replays](evidence/nicolas_symmetric_lut_public_4e30dcf_266c593/README.md), [three BF16 shape replays](evidence/nicolas_e4m3_lut_shapes_public_4f316ac_266c593/README.md), [E4M3 DIM16 requant](evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/README.md), [four E4M3 mesh requants](evidence/nicolas_e4m3_lut_requant_mesh_public_4553528_266c593/README.md), [two rectangular E4M3 requants](evidence/nicolas_e4m3_lut_requant_rect_public_75524c6_266c593/README.md), [E5M2 DIM16 requant](evidence/nicolas_e5m2_lut_requant_public_8e9706f_266c593/README.md), [E2M3 DIM16 requant](evidence/nicolas_e2m3_lut_requant_public_9df3a5e_266c593/README.md), [ten E2M3/E5M2 mesh requants](evidence/nicolas_e2m3_e5m2_lut_requant_mesh_public_b94c3af_266c593/README.md), [five E3M2 mesh requants](evidence/nicolas_e3m2_lut_requant_public_5b6dfc0_266c593/README.md) | Six named BF16 source programs and 24 packed LUT-index requant sources match through generated public objects on Spike. Other source families remain outside these suites. |
| FP8/FP4/FP6 matrix and quantized output | [Source shapes and precision cases](compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6), [requantizer modes](compiled_mx_pipeline.md#the-nicolas-requantizer-wrappers-three-output-modes), [FP8 I-chunk replays](evidence/nicolas_fp8_chunked_public_fbe26eb_266c593/README.md), [direct 2D-scale replay](evidence/nicolas_fp8_chunked_2d_public_287593c_266c593/README.md), [native DRAM loop](evidence/nicolas_fp8_native_dram_public_1ef6f85_266c593/README.md), [native N chunks](evidence/nicolas_fp8_native_nc_public_57b26ce_266c593/README.md), [loop-managed scales](evidence/nicolas_fp8_native_ls_public_126f138_266c593/README.md), [native K tiling](evidence/nicolas_fp8_native_kt_public_af0b4ad_266c593/README.md), [native scale controls](evidence/nicolas_fp8_native_scale_control_public_c2e25ce_266c593/README.md), [native store ReLU](evidence/nicolas_fp8_native_relu_public_c729276_266c593/README.md), [zero-base SMEM readout](evidence/nicolas_fp8_smem_zero_public_64ecacd_266c593/README.md), [three DRAM-mvout Spike fallback paths](evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593/README.md) | Selected source programs and full outputs pass. The three named I-chunk variants, nine native DRAM-loop variants, and three DRAM-mvout source fallback paths pass on Spike. The DRAM-mvout hardware accumulator paths and other transfer or performance variants remain unqualified. |
| Direct Nicolas FP8 source through public object compiler | [128³ typed object replay](evidence/nicolas_plain_fp8_typed_object_7d7a660_266c593/README.md) | The checked-in `matmul_tiled_fp8_128x128` packed arrays and all 16,384 BF16 source goldens match on pinned Spike. This covers the matrix result, not the C test's cache experiment or performance counters. |
| Additional direct Nicolas source shapes | [15-case suite](evidence/nicolas_direct_matrix_suite_9df3384_266c593/README.md), [three FP8 cases](evidence/nicolas_direct_fp8_three_789d192_266c593/README.md), [16 DIM8/16/32 cases](evidence/nicolas_direct_sixteen_28936e2_266c593/README.md), [alternate 32³ FP8 transfer source](evidence/nicolas_fp8_alternate_mvin_compiled_5a2dd35_266c593/README.md), [two I-chunk variants](evidence/nicolas_fp8_chunked_public_fbe26eb_266c593/README.md), [direct 2D-scale variant](evidence/nicolas_fp8_chunked_2d_public_287593c_266c593/README.md), [native DRAM loop](evidence/nicolas_fp8_native_dram_public_1ef6f85_266c593/README.md), [native N chunks](evidence/nicolas_fp8_native_nc_public_57b26ce_266c593/README.md), [loop-managed scales](evidence/nicolas_fp8_native_ls_public_126f138_266c593/README.md), [native K tiling](evidence/nicolas_fp8_native_kt_public_af0b4ad_266c593/README.md), [native scale controls](evidence/nicolas_fp8_native_scale_control_public_c2e25ce_266c593/README.md), [native store ReLU](evidence/nicolas_fp8_native_relu_public_c729276_266c593/README.md), [FP6 128³ repeated LUT](evidence/nicolas_plain_fp6_128_aliased_lut_public_6c6b27a_266c593/README.md), [zero-base SMEM readout](evidence/nicolas_fp8_smem_zero_public_64ecacd_266c593/README.md), [three DRAM-mvout Spike fallback paths](evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593/README.md) | Full-output replays cover 52 distinct source programs across FP8, FP4, FP6, irregular shapes, and DIM8/16/32 BF16 and requantized readout. The alternate and direct 2D-scale sources use already qualified objects after checked command-equivalence audits. The three DRAM-mvout source programs are qualified only for their Spike scratchpad fallback. |
| Deeper Nicolas FP8 through public object compiler | [128×128×256 two-wave replay](evidence/nicolas_plain_fp8_256_two_wave_object_9250250_266c593/README.md) | The checked-in source arrays and all 16,384 BF16 goldens match with a compiler-selected two-wave K schedule. This establishes numerical output parity, not exact source instruction or performance parity. |
| Irregular Nicolas FP8 through public object compiler | [96×96×64 replay](evidence/nicolas_plain_fp8_96x96x64_object_52dcc4d_266c593/README.md) | The checked-in source arrays and all 9,216 BF16 goldens match on pinned Spike. This qualifies one non-square geometry, not all rectangular programs. |
| Nicolas FP8 quantized readout through public object compiler | [128³ requant replay](evidence/nicolas_fp8_requant_typed_object_fa5ec73_266c593/README.md) | The checked-in `matmul_tiled_fp8_128x128_requant` source codes and scales match exactly: 16,384 FP8 codes and 512 E8M0 scales on pinned Spike. Other source quantization conventions require separate checks. |
| Nicolas DIM32 FP8 quantized readout through public object compiler | [64³ DIM32 requant replay](evidence/nicolas_fp8_dim32_requant_typed_object_835e5ba_266c593/README.md) | The checked-in `matmul_tiled_fp8_64x64_requant_dim32` codes and scales match exactly: 4,096 FP8 codes and 128 E8M0 scales on pinned Spike with the DIM32 extension. Other DIM32 variants still need qualification. |
| Direct Nicolas FP4 source through public object compiler | [64³ typed object replay](evidence/nicolas_plain_fp4_typed_object_97b0913_266c593/README.md) | The checked-in `matmul_tiled_fp4_64x64` direct packed arrays and all 4,096 BF16 source goldens match on pinned Spike. This qualifies the matrix result only. |
| Nicolas packed FP4 requantized readout through public object compiler | [64³ requant replay](evidence/nicolas_fp4_requant_typed_object_5657fb6_266c593/README.md) | The checked-in `matmul_tiled_fp4_64x64_requant` packed source output and scales match exactly: 2,048 packed bytes and 128 E8M0 scales on pinned Spike. Other FP4 requant variants still need qualification. |
| Nicolas larger FP4 requantized readout through public object compiler | [128×128×512 requant replay](evidence/nicolas_fp4_large_requant_typed_object_03490a5_266c593/README.md) | The checked-in `matmul_tiled_fp4_128x128x512_requant` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike. Other shapes and modes still need qualification. |
| Nicolas DIM32 FP4 requantized readout through public object compiler | [128³ DIM32 requant replay](evidence/nicolas_fp4_dim32_requant_typed_object_f475d05_266c593/README.md) | The checked-in `matmul_tiled_fp4_128x128_requant_dim32` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike with the DIM32 extension. Other DIM32 modes still need qualification. |
| Nicolas flat and tiled FP4 scratchpad requant through public object compiler | [64×128 dual-layout replay](evidence/nicolas_fp4_dual_public_object_d4ed0d8_266c593/README.md) | The pinned `spad_requant_fp4.c` source checker and compiler-linked driver each pass 16,384 FP4 code and 512 scale comparisons on Spike. The typed graph is generated from checked source geometry; it is not a model2MLIR capture. |
| Nicolas flat and tiled FP8 scratchpad requant through generated object | [32×64 dual-layout replay](evidence/nicolas_spad_requant_fp8_compiled_9d202b8_266c593/README.md) | The pinned `spad_requant.c` source checker and compiler-linked driver each pass 4,096 E4M3 code and 128 scale comparisons on Spike under both DIM16 VPU profiles. The typed graph is source-bound and the driver retains no handwritten accelerator commands. This synthetic C test has no PyTorch capture. |
| Direct Nicolas FP6 source through public object compiler | [128×128×512 typed object replay](evidence/nicolas_plain_fp6_typed_object_95fc6d5_266c593/README.md), [128³ repeated-LUT replay](evidence/nicolas_plain_fp6_128_aliased_lut_public_6c6b27a_266c593/README.md) | Both checked-in programs preserve their packed source arrays and all 64 A/B/C LUT lines; their generated objects each match all 16,384 BF16 source goldens on pinned Spike in serial mode. The older 128³ source has repeated LUT codes and uses a distinct structural capture witness. Alternating-buffer RTL timing is not qualified by these results. |
| Nicolas FP6 LUT-index requantized readout through public object compiler | [128×128×512 requant replay](evidence/nicolas_fp6_requant_typed_object_b25fa48_266c593/README.md) | The checked-in `matmul_tiled_fp6_128x128x512_requant` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike in serial mode. Other FP6 shapes and alternating-buffer RTL timing still need qualification. |
| VPU | [29 source checks](compiled_mx_pipeline.md#nicolas-vpu-source-oracle-across-all-operations), [14 public VPU objects](evidence/nicolas_vpu_public_objects_266c593/README.md), [full `vpu_ops` program replay](evidence/nicolas_vpu_ops_program_public_266c593/README.md), [public softmax object](evidence/mx_public_vpu_softmax_3f9af55/README.md) | The public compiler lowers all 14 base/fused VPU operation classes. A single generated object issues all 30 ordered `vpu_ops.c` VPU commands, including dependent operations, timed snapshots, and a write-after-read reload; 13,056 BF16 values match the source reference under both VPU Rocket profiles. The CPU input/reference flow remains a thin C driver, FP4 VPU operand semantics are not inferred from this BF16 replay, and RTL timing remains unqualified. |
| Legacy generic FP6 MMIO debug data | [Current model2MLIR Rocket replay](evidence/nicolas_ws_generic_portable_a042643_266c593/README.md) | Its source header's FP6 operands and 64-line LUT banks feed a compiler-generated Rocket object. All 16,384 BF16 values, 8,192 packed bytes, and 512 output scales match on Spike. The original fixed MMIO issue path and timing are unqualified; this is a numerical source-data result. |
| Connected MX/VPU/requant | [64³ resident FP8 chain](evidence/nicolas_plain_fp8_chain64_public_52e6132_266c593/README.md), [full connected chain](compiled_mx_pipeline.md#matrixvpurequant-source-chain), [pipelined source](compiled_mx_pipeline.md#source-preloaded-pipelined-issue-on-nicolass-spike) | The named 64³ source matches C1/C2 codes and scales through a public object on pinned Spike. Other checked graphs run; arbitrary source control flow and scheduling are not supported. |
| Bandwidth and microarchitectural tests | [Typed seven-phase MX memory benchmark replay](evidence/nicolas_mem_bw_typed_public_9cb0e4a_266c593/README.md), [single native DRAM-loop replay](evidence/nicolas_fp8_native_dram_public_1ef6f85_266c593/README.md), [two N-chunk loops](evidence/nicolas_fp8_native_nc_public_57b26ce_266c593/README.md), [two loop-managed-scale variants](evidence/nicolas_fp8_native_ls_public_126f138_266c593/README.md), [native K tiling](evidence/nicolas_fp8_native_kt_public_af0b4ad_266c593/README.md), [two scale-control variants](evidence/nicolas_fp8_native_scale_control_public_c2e25ce_266c593/README.md), [native store ReLU](evidence/nicolas_fp8_native_relu_public_c729276_266c593/README.md), and other matrix DRAM-loop variants | The nine native-loop programs match all BF16 outputs and command operands on Spike. `mx_mem_bw` lowers typed transfer ops and matches seven source phase geometries and the full 16 KiB readout. Benchmark timing parity and performance counters remain unqualified. |

The [31-driver Radiance replay](evidence/radiance_mx_gemm_f193c8f_80f84ca/README.md)
is a different roster. It proves the current Radiance MX GEMM sources, not all
171 Nicolas MX tests. A complete Nicolas-source claim would require an explicit
per-program inventory, a typed capture or declared equivalent input for each
program, a generated executable, and a full-output comparison against its
source oracle under the appropriate profile. Performance-only tests would
also require comparable counter and timing criteria.
