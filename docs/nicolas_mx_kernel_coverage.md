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
exact hash. **152 programs have at least one direct source-hash reference; 19
have none.** A matching hash establishes provenance only. Some referenced
receipts are source-oracle runs or frontend captures, so neither group should
be read as a count of compiler-regenerated programs. Rebuild the inventory
with `python -m tools.audit_nicolas_mx_roster --rtl-root "$MX_RTL_ROOT"
--out docs/evidence/nicolas_mx_source_inventory_266c593/index.json --check`.

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
The [three same-format LUT source programs](evidence/nicolas_symmetric_lut_public_4e30dcf_266c593/README.md)
also pass through public data-free objects with their dedicated E2M3, E4M3,
and E5M2 Rocket profiles, matching 12,288 BF16 values. Three more
[E4M3 LUT shape and mesh replays](evidence/nicolas_e4m3_lut_shapes_public_4f316ac_266c593/README.md)
match 24,576 BF16 values on DIM8 and DIM32. Together, these exhaustive
suites cover the numerical results of **117 distinct named MX
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
suites cover **141 of 171 named MX source programs**. The separate source
program qualifications below are outside that matrix/LUT count.
The [earlier 11-case suite](evidence/nicolas_direct_matrix_suite_9f3a759_266c593/README.md)
remains archived. The 152 source-hash references above include many weaker
forms of evidence and are not a count of compiler-regenerated programs.

| Source family | Compiler evidence | Coverage boundary |
|---|---|---|
| Asymmetric matrix modes | [74-program public-object suite](evidence/nicolas_asym_public_suite_e6923e8_266c593/README.md), [mode matrices](compiled_mx_pipeline.md#dedicated-dim16-asymmetric-mode-matrix), and [stock-model gap](compiled_mx_pipeline.md#isolated-weight-lut-spike-correction-across-all-legal-modes) | All 74 pinned `matmul_tiled_asym_*` C programs match 360,960 BF16 source outputs through generated public objects on Spike. This is named-program numerical coverage; separate legal-mode tests still expose the stock Spike gap, and no FPGA timing parity is claimed. |
| Same-format LUT E2M3/E3M2/E4M3/E5M2 | [Three BF16 mode replays](evidence/nicolas_symmetric_lut_public_4e30dcf_266c593/README.md), [three BF16 shape replays](evidence/nicolas_e4m3_lut_shapes_public_4f316ac_266c593/README.md), [E4M3 DIM16 requant](evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/README.md), [four E4M3 mesh requants](evidence/nicolas_e4m3_lut_requant_mesh_public_4553528_266c593/README.md), [two rectangular E4M3 requants](evidence/nicolas_e4m3_lut_requant_rect_public_75524c6_266c593/README.md), [E5M2 DIM16 requant](evidence/nicolas_e5m2_lut_requant_public_8e9706f_266c593/README.md), [E2M3 DIM16 requant](evidence/nicolas_e2m3_lut_requant_public_9df3a5e_266c593/README.md), [ten E2M3/E5M2 mesh requants](evidence/nicolas_e2m3_e5m2_lut_requant_mesh_public_b94c3af_266c593/README.md), [five E3M2 mesh requants](evidence/nicolas_e3m2_lut_requant_public_5b6dfc0_266c593/README.md) | Six named BF16 source programs and 24 packed LUT-index requant sources match through generated public objects on Spike. Other source families remain outside these suites. |
| FP8/FP4/FP6 matrix and quantized output | [Source shapes and precision cases](compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6), [requantizer modes](compiled_mx_pipeline.md#the-nicolas-requantizer-wrappers-three-output-modes), [FP8 I-chunk replays](evidence/nicolas_fp8_chunked_public_fbe26eb_266c593/README.md) | Selected source programs and full outputs pass. The two named I-chunk variants pass on Spike; other DRAM-loop, transfer, and performance variants are not exhaustively reproduced from typed MLIR. |
| Direct Nicolas FP8 source through public object compiler | [128³ typed object replay](evidence/nicolas_plain_fp8_typed_object_7d7a660_266c593/README.md) | The checked-in `matmul_tiled_fp8_128x128` packed arrays and all 16,384 BF16 source goldens match on pinned Spike. This covers the matrix result, not the C test's cache experiment or performance counters. |
| Additional direct Nicolas source shapes | [15-case suite](evidence/nicolas_direct_matrix_suite_9df3384_266c593/README.md), [three FP8 cases](evidence/nicolas_direct_fp8_three_789d192_266c593/README.md), [16 DIM8/16/32 cases](evidence/nicolas_direct_sixteen_28936e2_266c593/README.md), [alternate 32³ FP8 transfer source](evidence/nicolas_fp8_alternate_mvin_compiled_5a2dd35_266c593/README.md), [two I-chunk variants](evidence/nicolas_fp8_chunked_public_fbe26eb_266c593/README.md) | Full-output replays cover 37 distinct source programs across FP8, FP4, FP6, irregular shapes, and DIM8/16/32 BF16 and requantized readout. The alternate source uses the same generated object after a checked transfer equivalence audit. |
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
| Direct Nicolas FP6 source through public object compiler | [128×128×512 typed object replay](evidence/nicolas_plain_fp6_typed_object_95fc6d5_266c593/README.md) | The checked-in `matmul_tiled_fp6_128x128x512` packed arrays, all 64 LUT lines for A/B/C, and all 16,384 BF16 source goldens match on pinned Spike in serial mode. The alternating-buffer RTL schedule is not qualified by this result. |
| Nicolas FP6 LUT-index requantized readout through public object compiler | [128×128×512 requant replay](evidence/nicolas_fp6_requant_typed_object_b25fa48_266c593/README.md) | The checked-in `matmul_tiled_fp6_128x128x512_requant` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike in serial mode. Other FP6 shapes and alternating-buffer RTL timing still need qualification. |
| VPU | [29 source checks](compiled_mx_pipeline.md#nicolas-vpu-source-oracle-across-all-operations), [compiler-issued operations](compiled_mx_pipeline.md#compiler-issued-base-vpu-operations), [public softmax object](evidence/mx_public_vpu_softmax_3f9af55/README.md) | All named VPU checks have compiler-issued counterparts, and softmax runs through the public object CLI. The entire `vpu_ops.c` control flow is not compiled as one MLIR program. |
| Connected MX/VPU/requant | [Full connected chain](compiled_mx_pipeline.md#matrixvpurequant-source-chain), [pipelined source](compiled_mx_pipeline.md#source-preloaded-pipelined-issue-on-nicolass-spike) | Specific checked graphs run; arbitrary source control flow and scheduling are not supported. |
| Bandwidth and microarchitectural tests | Source `mx_mem_bw.c` and matrix DRAM-loop variants | The compiler has physical transfer schedules, but no claim of complete source-program or performance parity for these benchmarks. |

The [31-driver Radiance replay](evidence/radiance_mx_gemm_f193c8f_80f84ca/README.md)
is a different roster. It proves the current Radiance MX GEMM sources, not all
171 Nicolas MX tests. A complete Nicolas-source claim would require an explicit
per-program inventory, a typed capture or declared equivalent input for each
program, a generated executable, and a full-output comparison against its
source oracle under the appropriate profile. Performance-only tests would
also require comparable counter and timing criteria.
