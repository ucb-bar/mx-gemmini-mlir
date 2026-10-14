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
| `attn_vpu_mid.c` | Same dense QK → VPU → resident requant → PV schedule at 64×128×64 | The [source baseline](evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/README.md) passes twice on Spike; clean model2MLIR `a042643` captures both differently shaped contractions. The [typed program](evidence/nicolas_attn_vpu_mid_typed_program_a042643_0344bcb/README.md) lowers all 302 MX commands plus one fence to a data-free object. Linked with Nicolas's checker, all 2,048 score words and four stage hashes match twice on Spike. | Extend the source-bound path to model-derived attention heads, including their scale and mask policies. |
| `attn_vpu_fa.c` | Same dense schedule at 64×256×128, with a different scale scalar and larger resident score | The [source baseline](evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/README.md) passes twice on Spike. The [typed program](evidence/nicolas_attn_vpu_fa_typed_program_a042643_0344bcb/README.md) lowers all 702 MX commands plus one fence to a data-free object. Linked with Nicolas's checker, all 4,096 score words and four stage hashes match twice on Spike. | Extend this dense lowering to model-derived heads and their scale and mask policies. |
| `llama_attention_full.c` | Q/K/V projections, MX QK, **Rocket f32 softmax**, requantized resident PV, `o_proj`, all 32 heads | One reduced-model head's QK, VPU softmax, and BF16 PV output are checked separately; model2MLIR worklists identify the contractions. | Generate a full attention graph with the source's host-softmax option or a separately declared VPU policy; check every head, requantized PV codes/scales, resident `o_proj`, and final output. |
| `attn_flash.c` and `attn_flash_*` | Blocked and fused attention variants, including E2M1 FP4 experiments | The [64×128×64 `attn_flash_mid` program](evidence/nicolas_flash_mid_typed_a042643_0344bcb/README.md) and [64×256×128 default program](evidence/nicolas_flash_fa_typed_a042643_0344bcb/README.md) lower complete source-bound typed graphs through the public object compiler. Their data-free serial objects match all 4,096 and 8,192 BF16 outputs, respectively, against Nicolas's pipelined source on two Spike runs. A [small FP4 configuration](evidence/nicolas_flash_mid_fp4_typed_a042643_0344bcb/README.md), the [2,048-key streamed variant](evidence/nicolas_flash_llama_stream_typed_e9ded36_0344bcb/README.md), its [causal-mask variant](evidence/nicolas_flash_llama_causal_stream_typed_e9ded36_0344bcb/README.md), the [uneven-block variant](evidence/nicolas_flash_llama_vb_stream_typed_e9ded36_0344bcb/README.md), and [two-V-buffer variant](evidence/nicolas_flash_llama_wide_stream_typed_e9ded36_0344bcb/README.md) also have full-output Spike parity for their selected source data. | Qualify fused VPU, larger FP4, and real model-head variants as distinct source policies and compare full outputs on Spike. |
| `llama_mlp*`, `mlp_proj_fp4*`, `llama_layer*`, `llama_model*` | Multiple projections, normalization, activation, residual, and layer/model control | Captured TinyLlama prefill and decode now link as one RV64 MX+VPU session with 200 selected MX operations in each stage; its 33-step Spike run is in progress. This compiles the captured graph, not the named fused C programs. | Complete TinyLlama's 33-step output and cycle check; then compare complete source MLP, layer, and fused MX+VPU schedules on the same captured data. |
| `mx_bench_matmul*`, `mx_fp4_dramloop.c`, `mxl*` | Native loops, tiling, utilization, and diagnostic variants | MX physical objects cover selected native-loop and ISA test paths. This application roster has no per-program compiler receipt. | Map each named source's profile and data header to a generated object, then compare the complete source oracle and the issued schedule where performance behavior is the subject. |

Separately, the [full 12-step SmolVLA session](evidence/nicolas_matrix_substitution_smolvla_full_20261013/README.md)
completes with equal outputs when Nicolas's MX matrix issuers replace the
two compiler catalogs. Its host and VPU objects are held fixed, so this is
not a stitched source implementation of the entire model.

The earlier TinyLlama head evidence is numerical and source-bound, but it uses
host staging between three ELFs. The dense `attn_vpu.c` program qualifies an
on-chip QK→VPU→PV sequence for its 32×64×64 source shape. The later linked
sessions establish full captured-graph compilation, while the issuer-swap
baseline leaves their host and VPU code fixed. None of these results yet
establishes equivalence to the resident `P @ V → o_proj` path in
`llama_attention_full.c` on real model inputs. The application source
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
