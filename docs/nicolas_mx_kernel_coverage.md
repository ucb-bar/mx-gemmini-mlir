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

| Source family | Compiler evidence | Coverage boundary |
|---|---|---|
| Asymmetric matrix modes | [DIM8/16/32 mode matrices](compiled_mx_pipeline.md#dedicated-dim16-asymmetric-mode-matrix) and [stock-model gap](compiled_mx_pipeline.md#isolated-weight-lut-spike-correction-across-all-legal-modes) | Mode-class tests pass 35 of 36 legal cells on stock Spike per mesh. This does not regenerate every named C program or variant. |
| FP8/FP4/FP6 matrix and quantized output | [Source shapes and precision cases](compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6), [requantizer modes](compiled_mx_pipeline.md#the-nicolas-requantizer-wrappers-three-output-modes) | Selected source programs and full outputs pass. Chunked, DRAM-loop, transfer, and performance variants are not exhaustively reproduced from typed MLIR. |
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
