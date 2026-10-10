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
exact hash. **98 programs have at least one direct source-hash reference; 73
have none.** A matching hash establishes provenance only. Some referenced
receipts are source-oracle runs or frontend captures, so neither group should
be read as a count of compiler-regenerated programs. Rebuild the inventory
with `python -m tools.audit_nicolas_mx_roster --rtl-root "$MX_RTL_ROOT"
--out docs/evidence/nicolas_mx_source_inventory_266c593/index.json --check`.

The [one-command direct matrix suite](evidence/nicolas_direct_matrix_suite_9f3a759_266c593/README.md)
reran all **11** registered direct matrix cases through fresh model2MLIR
capture, public object compilation, and full-output Spike checks. Its index
records 111,872 comparisons across BF16 values, quantized bytes, and scales.
These 11 cases are the fully replayed direct matrix subset; the 98 source-hash
references above include many weaker forms of evidence.

| Source family | Compiler evidence | Coverage boundary |
|---|---|---|
| Asymmetric matrix modes | [DIM8/16/32 mode matrices](compiled_mx_pipeline.md#dedicated-dim16-asymmetric-mode-matrix) and [stock-model gap](compiled_mx_pipeline.md#isolated-weight-lut-spike-correction-across-all-legal-modes) | Mode-class tests pass 35 of 36 legal cells on stock Spike per mesh. This does not regenerate every named C program or variant. |
| FP8/FP4/FP6 matrix and quantized output | [Source shapes and precision cases](compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6), [requantizer modes](compiled_mx_pipeline.md#the-nicolas-requantizer-wrappers-three-output-modes) | Selected source programs and full outputs pass. Chunked, DRAM-loop, transfer, and performance variants are not exhaustively reproduced from typed MLIR. |
| Direct Nicolas FP8 source through public object compiler | [128³ typed object replay](evidence/nicolas_plain_fp8_typed_object_7d7a660_266c593/README.md) | The checked-in `matmul_tiled_fp8_128x128` packed arrays and all 16,384 BF16 source goldens match on pinned Spike. This covers the matrix result, not the C test's cache experiment or performance counters. |
| Deeper Nicolas FP8 through public object compiler | [128×128×256 two-wave replay](evidence/nicolas_plain_fp8_256_two_wave_object_9250250_266c593/README.md) | The checked-in source arrays and all 16,384 BF16 goldens match with a compiler-selected two-wave K schedule. This establishes numerical output parity, not exact source instruction or performance parity. |
| Irregular Nicolas FP8 through public object compiler | [96×96×64 replay](evidence/nicolas_plain_fp8_96x96x64_object_52dcc4d_266c593/README.md) | The checked-in source arrays and all 9,216 BF16 goldens match on pinned Spike. This qualifies one non-square geometry, not all rectangular programs. |
| Nicolas FP8 quantized readout through public object compiler | [128³ requant replay](evidence/nicolas_fp8_requant_typed_object_fa5ec73_266c593/README.md) | The checked-in `matmul_tiled_fp8_128x128_requant` source codes and scales match exactly: 16,384 FP8 codes and 512 E8M0 scales on pinned Spike. Other source quantization conventions require separate checks. |
| Nicolas DIM32 FP8 quantized readout through public object compiler | [64³ DIM32 requant replay](evidence/nicolas_fp8_dim32_requant_typed_object_835e5ba_266c593/README.md) | The checked-in `matmul_tiled_fp8_64x64_requant_dim32` codes and scales match exactly: 4,096 FP8 codes and 128 E8M0 scales on pinned Spike with the DIM32 extension. Other DIM32 variants still need qualification. |
| Direct Nicolas FP4 source through public object compiler | [64³ typed object replay](evidence/nicolas_plain_fp4_typed_object_97b0913_266c593/README.md) | The checked-in `matmul_tiled_fp4_64x64` direct packed arrays and all 4,096 BF16 source goldens match on pinned Spike. This qualifies the matrix result only. |
| Nicolas packed FP4 requantized readout through public object compiler | [64³ requant replay](evidence/nicolas_fp4_requant_typed_object_5657fb6_266c593/README.md) | The checked-in `matmul_tiled_fp4_64x64_requant` packed source output and scales match exactly: 2,048 packed bytes and 128 E8M0 scales on pinned Spike. Other FP4 requant variants still need qualification. |
| Nicolas larger FP4 requantized readout through public object compiler | [128×128×512 requant replay](evidence/nicolas_fp4_large_requant_typed_object_03490a5_266c593/README.md) | The checked-in `matmul_tiled_fp4_128x128x512_requant` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike. Other shapes and modes still need qualification. |
| Nicolas DIM32 FP4 requantized readout through public object compiler | [128³ DIM32 requant replay](evidence/nicolas_fp4_dim32_requant_typed_object_f475d05_266c593/README.md) | The checked-in `matmul_tiled_fp4_128x128_requant_dim32` packed source output and scales match exactly: 8,192 packed bytes and 512 E8M0 scales on pinned Spike with the DIM32 extension. Other DIM32 modes still need qualification. |
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
