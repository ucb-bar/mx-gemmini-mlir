# Nicolas MX application kernels: compiler coverage

The [current application source](https://github.com/ucb-bar/npu-exploration/tree/main/baremetal/mxgemmini/src)
is pinned here to `npu-exploration` `0344bcba70b899039ccd0ac0801ca5605d46d3eb`
(`baremetal/mxgemmini/Makefile` SHA-256
`4b745d38a90775de84aa7eadab0768cf905f37417c3c6afb418c51cc7184dc2a`).
Its `KERNELS` variable names **63 application and benchmark programs**, with
21 `mxl0`…`mxl20` diagnostic rungs and separate model targets. This is a
different roster from the 171 programs in Nicolas's Gemmini ISA test Makefile;
[their compiler audit](nicolas_mx_kernel_coverage.md) does not establish
regeneration of these application ELFs.

| Source path | Source dataflow | Compiler evidence | Next executable gate |
|---|---|---|---|
| `attn_vpu.c` | MX QK → seven VPU scale/softmax ops → resident E4M3 requantization → MX PV | The [source baseline](evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/README.md) passes the unmodified dense 32×64×64 C kernel on Spike and archives a current model2MLIR attention graph. The [typed program](evidence/nicolas_attn_vpu_typed_program_e9ded36_0344bcb/README.md) lowers scale and operand DMA, QK, VPU, resident requant, PV, and readouts through one public-CLI object. All 127 command items match the source-equivalent stream; all four source stage hashes match twice on Spike. The [TinyLlama head replay](evidence/model2mlir_tinyllama_connected_attention_head_a042643_266c593/README.md) checks model-derived QK, host scaling/masking, VPU softmax, and PV in separate Spike programs. | Generalize the typed lowering to other attention shapes and model-derived head layouts; qualify full heads and models against source data. |
| `attn_vpu_mid.c` | Same dense QK → VPU → resident requant → PV schedule at 64×128×64 | The [source baseline](evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/README.md) passes twice on Spike; clean model2MLIR `a042643` captures both differently shaped contractions. The [typed program](evidence/nicolas_attn_vpu_mid_typed_program_a042643_0344bcb/README.md) lowers all 302 MX commands plus one fence to a data-free object. Linked with Nicolas's checker, all 2,048 score words and four stage hashes match twice on Spike. | Extend the shape plan to `attn_vpu_fa.c` and model-derived attention heads, including their scale and mask policies. |
| `llama_attention_full.c` | Q/K/V projections, MX QK, **Rocket f32 softmax**, requantized resident PV, `o_proj`, all 32 heads | One reduced-model head's QK, VPU softmax, and BF16 PV output are checked separately; model2MLIR worklists identify the contractions. | Generate a full attention graph with the source's host-softmax option or a separately declared VPU policy; check every head, requantized PV codes/scales, resident `o_proj`, and final output. |
| `attn_flash.c` and `attn_flash_*` | Blocked and fused attention variants, including E2M1 FP4 experiments | Individual matrix, VPU, and requant command classes have Spike receipts; no complete flash-attention source program has a compiler-generated execution receipt. | Capture or construct a verified blocked attention graph, lower its running state and barriers, and compare a complete source kernel on Spike. |
| `llama_mlp*`, `mlp_proj_fp4*`, `llama_layer*`, `llama_model*` | Multiple projections, normalization, activation, residual, and layer/model control | Model2MLIR exports and selected projection objects exist; full layer and model graphs do not lower to an executable. | Lower a complete source MLP, then one decoder layer and model, with explicit FP4 regular-matmul and FP8 attention policies and source-data checks. |
| `mx_bench_matmul*`, `mx_fp4_dramloop.c`, `mxl*` | Native loops, tiling, utilization, and diagnostic variants | MX physical objects cover selected native-loop and ISA test paths. This application roster has no per-program compiler receipt. | Map each named source's profile and data header to a generated object, then compare the complete source oracle and the issued schedule where performance behavior is the subject. |

The TinyLlama head evidence is numerical and source-bound, but it uses host
staging between three ELFs. The dense `attn_vpu.c` program qualifies an
on-chip QK→VPU→PV sequence for its 32×64×64 source shape. Neither result
establishes the resident `P @ V → o_proj` path in `llama_attention_full.c` or
full-model compilation. The application source
also replans K tiles and output chunks from scratchpad capacity; compiler
qualification must exercise that profile-dependent scheduling. The source
readme records 8,192-row and 16,384-row scratchpad plans for its full
attention kernel.

The first reduced TinyLlama head is 8×8. Its current 16×32 QK/PV padding
contains only **16 E8M0 blocks** (`M × N / 32`), while the selected
`spad_requant` command requires a multiple of 32 blocks. The compiler
correctly rejects requantization of that shape. A single on-device head
therefore needs at least **32×32** score and probability storage, with
the masked and padded lanes checked numerically, or a validated head-packing
plan. Merely concatenating the three current objects would omit this
necessary layout change.

All gates in this table are compiler, source, and Spike checks. FPGA execution
is outside the current qualification scope.
