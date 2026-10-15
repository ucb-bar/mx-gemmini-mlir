# MX Gemmini dialect overview slides

[Editable PowerPoint deck](mx_gemmini_dialect_overview.pptx) contains four
slides based on the supplied Lab Pastel PowerPoint template. The diagrams,
text, callout, and results table are PowerPoint objects. The university seal
is the template's master image. The MLIR, C, and JSON examples use editable
syntax-colored text runs.

## Slide 1: compiler structure

The C++ dialect defines typed MX resource, contraction, readout, VPU,
requantization, and ordering operations in
[`MxOps.td`](../../include/MxGemmini/MxOps.td). `mx-gemmini-opt` verifies this
IR. Profile and payload checks, tile planning, physical commands, and Merlin
integration currently live in Python modules under `mx_gemmini_support/`.
`MXBF16CatalogPreparation` and `MXVPUPostOffload` are Merlin-side
transformations, not registered C++ MLIR passes.

Slide 1 quotes an abbreviated form of the FP4 payload in
[`payload_bound.mlir`](../evidence/nicolas_direct_sixteen_28936e2_266c593/fp4_64x64x64_nonrequant_dim32/payload_bound.mlir).
Its four `mx_gemmini.resource` operands contain packed activation/weight
codes and E8M0 scales. `mx_gemmini.contract` fixes the FP4 direct PE tuple;
it requests the matrix reduction `C[m,n] = Σk A[m,k]×B[k,n]` in that mode.
`mx_gemmini.readout_bf16` requests the result. The physical lowering produces
`mvin`, MX compute, VPU, and `mvout` commands as needed by the selected
kernel. The right code box quotes the first `mvin` from the generated
[`mx_issue.c`](../evidence/attn_vpu_equal_work_20261013/object/timed/mx_issue.c)
for the dense attention object. Its compiled RV64 object contains
`04b6b07b` at offset `0x4a` (`.insn 4, 0x04b6b07b`). That is one actual
RoCC instruction word, not the whole program binary.

The five named transformations/checks on the slide have distinct jobs:

| Stage | Concrete effect |
| --- | --- |
| `bind_profile` | Replaces the captured `format = "mxfp4"` with a legal PE tuple and stamps the exact profile digest on the MX ops. |
| `MXBF16CatalogPreparation` | Selects eligible BF16 matmuls in the Merlin graph, inserts the needed f32 boundary, and records the quantization choice. |
| `schedule.plan` | Binds a catalog symbol to `mx_native_dram_loop` or `mx_serial_commands`; issuer lowering consumes the selected policy. |
| `MXVPUPostOffload` | Rewrites source-bound softmax, add, elementwise, unary, and reduction regions to VPU calls. |
| `physical_program` | Orders DMA, MX, VPU, wait, and readback commands before C/assembly emission. |

The mapping to the reference kernels is explicit: Radiance FP4/FP8 GEMMs
correspond to `resource` + `contract` + readout. Compiler-side vector work
can use `vpu_execute`, `spad_requant`, and `wait`; Radiance's mixed flash
attention source instead runs softmax on Muon SIMT. The slide shows a
representative path, while each source kernel's selected precision and tile
shape still come from its captured workload and target profile.

The standalone command path is typed MX MLIR → physical RoCC commands →
generated C issuer and assembly data → RV64 GCC → ELF. It bypasses LLVM.
The full-model path currently lowers the generic host graph through Merlin
MLIR → LLVM IR → RV64 while using MX command objects for accelerated regions.
See [compiler path](../compiled_mx_pipeline.md) and
[full-model compilation](../compile_e2e_models.md).

## Slide 2: configuration model

The profile exporter projects 41 Gemmini MX fragments and 40 Chipyard
wrappers from the pinned `gemmini-mx-cleanup` RTL and MxGen revision. JSON
profiles carry geometry, memory, queue, scale/LUT, legal PE tuple, and VPU
capabilities. They are structural descriptions; a legal tuple still needs
numerical and timing qualification on its selected target. The selected
`MxE4M3Fp4VpuGemminiRocketConfig` has direct E4M3 and FP4 modes, two
8-lane BF16 VPUs, and `SPAD_REQUANT`. Its operand LUT is absent (`lut: false`);
FP6 is available in a separate
non-VPU configuration. Nicolas's `npu-exploration` kernels provide source
checks and performance reference programs, while RTL defines hardware
capabilities. See [profile inventory](../current_mx_profiles.md).

Here a hardware mode is a legal *tuple*: activation format, weight format,
direct or LUT projection for each operand, and the PE mode number. The number
does not identify a precision on its own. This selected MX+VPU profile has
exactly two matrix tuples: PE 0 is direct FP4 E2M1 × FP4 E2M1, and PE 8 is
direct FP8 E4M3 × FP8 E4M3. VPU opcodes are separate from these matrix
tuples. The 81 projected profiles comprise 41 Gemmini fragments and 40
Chipyard wrappers, not 81 independently performance-qualified systems.

The JSON box is a shortened excerpt of the selected
[`MxE4M3Fp4VpuGemminiRocketConfig` profile](../../profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json).
For this file, `profile_sha256(profile)` is
`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`.
The [`bind_handoff`](../../mx_gemmini_support/bind_profile.py) code selects
the one direct FP4 row for a captured `mxfp4` contract and emits
`activation_format = "fp4_e2m1"`, `weight_format = "fp4_e2m1"`,
`pe_mode = 0`, and that digest. With this profile an FP6 contract has no
legal row and fails binding. The profile's 256 KiB scratchpad and 16×16
mesh also constrain the later tile plan. Profile binding checks the pinned
RTL hashes when invoked with `--rtl-root`.

Switching to the
[`MxAsymFp6Fp4GemminiRocketConfig` profile](../../profiles/gemmini-mx-cleanup-266c593/MxAsymFp6Fp4GemminiRocketConfig.json)
changes the legal matrix row to FP6 E3M2 with an activation LUT times direct
FP4 E2M1 at PE mode 3. That profile has a LUT and no VPU. This demonstrates
why the compiler checks the full tuple rather than interpreting `pe_mode`
alone: modes 0–8 can also use LUT-projected operands; modes 9–11 add the
four-product LUT path. The two examples are separate SoC configurations.

The captured TinyLlama prefill and decode stage catalogs each have 200 MX
`bindings`: 155 FP4 and 45 FP8, counted from each binding's
`target_dtypes`. The prepacked catalog has 158 distinct `kernels` because
some bindings share an issuer; kernel count is not operation count. The
[full-model compilation guide](../compile_e2e_models.md) describes how to
regenerate the stage catalogs and inspect their precision choices.

## Slide 3: measured scope

The rows answer different questions. They must not be read as one
whole-model compiler-versus-handwritten speedup:

The full SmolVLA row uses the
[all-FP8 issuer-swap comparison](../evidence/nicolas_matrix_substitution_smolvla_all_fp8_20261014/README.md).
Both complete runs have 33 matching output hashes. The generator binds
the compiler timing to the archived all-FP8 full-action run. The earlier
mixed FP4/FP8 issuer-swap result remains a separate control.

The independent 160-token TinyLlama source forward has completed at
12,703,713,784 modeled cycles with matched IDs and checkpoint tensors.
The matched compiler one-pass takes 15,170,996,572 cycles. Their late
top tokens match at 24/33 positions. The table displays both raw counts
and **NO RATIO** because the source starts after embedding lookup and the
host/VPU placement differs. This row is a different program boundary from
the cached 33-step session above it.

| Row | Changed work | What the result tests |
| --- | --- | --- |
| Full TinyLlama and SmolVLA sessions | MX matrix issuers only; host graph and VPU objects stay fixed | MX command emission and its cost inside the captured program |
| TinyLlama independent 160-token forward | Nicolas's complete 22-layer source program versus the matched compiler one-pass | Raw timing and late top-token agreement across different host/VPU boundaries |
| Captured SmolVLA attention | One final-step head at each of 113 and 163 keys; 4,096 BF16 values each | Pipelined source versus compiler cross-block scheduling on actual model operands |

The [final-step paired archive](../evidence/smolvla_attention_session_20261014/README.md)
uses `e4m3_vpu` for both sides: compiler/source cycles are 14,132/15,165
for 113 keys and 15,733/16,956 for 163 keys. Both pairs match every BF16
output. This is a head-level comparison; the full SmolVLA issuer-swap row
keeps the host graph and VPU objects fixed.

The earlier [64×256×128 dense-attention result](../evidence/attn_vpu_equal_work_20261013/README.md)
remains a separate kernel control. The slide now shows captured SmolVLA
attention heads from the final flow step. A Merlin `schedule.plan` selects
the compiler's cross-block pipeline, which runs −6.8%/−7.2% against the
same guest's pipelined C source for those two heads. A compiler-first cold
check matches all BF16 outputs. Earlier first-call controls under the
extension's default timing model found −4.1%/−3.1% for the pipelined graph,
+5.7%/+9.9% for queued, and +10.1%/+23.2% for fenced. Those settings
and operands differ from the final-step row; all are head-level measurements.

The requested complete-session performance comparison needs independently
written MX+VPU programs with matched inputs, weights, precision, outputs,
and timing boundaries. Neither full-session issuer swap supplies that
program. Nicolas's all-FP8 160-token TinyLlama forward is independent and
uses the same IDs and checkpoint tensors as the compiler one-pass capture;
the compiler Spike run now completes with 24/33 matching late top tokens.
It is not the cached 33-step session, and its source timer excludes the
compiler's embedding lookup. There is no valid full-session compiler/source
speedup yet.

The [prefill precision archive](../evidence/tinyllama_prefill_precision_20261014/README.md)
adds a separate numerical diagnostic on the captured FP32 input: the mixed
FP4/FP8 compiler prefill has logit cosine −0.313, while an all-FP8 compiler
prefill has cosine 0.943. Neither selects the FP32 top token. The mixed
prefill bytes match the completed full-session compiler output hash; the
completed [33-step logit archive](../evidence/tinyllama_full_logits_20261014/README.md)
confirms all 99 output hashes. Mean decode cosine to captured FP32 is −0.324
with 0/32 top-token matches. The slide presents these values separately
from issuer-swap cycle comparisons.
Promoting the first 11 layers' 77 dense sites to FP8 in a separate
[Spike prefill](../evidence/tinyllama_prefill_precision_20261014/first11_prepacked_spike_receipt.json)
gives cosine −0.440. Its prepacked and runtime-packed forms have identical
logits; modeled prefill cycles are 12,114,181,400 and 90,552,028,722,
respectively. This is a packing-cost comparison for one prefill, not a
full-session compiler-versus-source speedup.

The [original SmolVLA action archive](../evidence/smolvla_full_action_values_20261014/README.md)
checks the final action and all ten flow outputs against the captured
unquantized FP32 trajectory. Its flow-step cosine falls from 0.997 at step 0
to 0.054 at step 9/final action; final normalized RMSE is 1.726. All 33
output digests match the earlier timed compiler run. The output-tap binary
has different stage objects and takes 613,171,131,189 modeled cycles; that
cycle count is excluded from the issuer-swap row.

The [corrected full SmolVLA archive](../evidence/smolvla_scalar_full_action_20261014/receipt.json)
keeps the same checkpoint, profile, prefix stage, and action stage while
updating the flow stage's scalar timestep through FP32. Its ten timestep
outputs match the FP32 recurrence exactly. The final action cosine is
**0.058** versus **0.054** for the original route; normalized RMSE is
**1.830**. The corrected output-tap run takes 316,515,603,374 modeled
cycles. Both output-tap runs use the same program roster and taps, and the
prefix and action step cycles are unchanged. Replacing the one-element VPU
path lowers each flow step from about 41.13 billion to 11.47 billion
modeled cycles. The output values differ, so that is not an equal-output
speedup; the fast issuer-swap row also has a different timing boundary.
Correcting the timestep alone does not remove the accumulated action drift.

The [full all-FP8 SmolVLA control](../evidence/smolvla_all_fp8_full_action_20261014/README.md)
binds all 391 selected prefix and flow contractions to direct E4M3. It
preserves the corrected run's captured programs, checkpoint weight blob,
target profile, arena layout, and output ABI. All 12 steps complete on
Nicolas's MX+VPU Spike. Final action cosine to captured FP32 rises from
0.058102 (mixed) to **0.988524** (all-FP8), while normalized RMSE falls
from 1.830469 to **0.246502**. The complete session takes
180,219,637,915 modeled cycles versus 316,515,603,374 for the corrected
mixed control. Their outputs differ, so this is a precision-policy cost
comparison, not an equal-output speedup or a source-kernel baseline.
The matched all-FP8 Nicolas issuer control takes 179,451,403,096 cycles
for the same 12 steps and outputs, making the compiler's MX issuer overhead
0.428% within the fixed host/VPU program.

The [full all-FP8 TinyLlama control](../evidence/tinyllama_all_fp8_full_session_20261014/README.md)
completes the 128-token prefill and all 32 cached decodes in
72,901,458,175 modeled cycles. Against the captured FP32 logits, prefill
cosine is 0.774 and mean decode cosine is 0.663, with 2/32 matching decode
top tokens. Slide 3 places these beside the mixed FP4/FP8 numerical results;
the full-session issuer-swap row still uses the mixed policy. Their outputs
differ, so their cycle totals do not form an equal-output speedup.

The separate [VPU route control](../evidence/tinyllama_fp8_vpu_route_control_20261014/README.md)
keeps the 200 FP8 matrix bindings and standalone timer boundary fixed.
The safe add/reduce/unary route takes 10,804,276,937 cycles; adding VPU
softmax and elementwise takes 12,276,460,508 cycles. Their output hashes
and top tokens differ, so the measured cycle difference is not an
equal-output speedup. The results table stays focused on issuer swaps and
the independent TinyLlama source forward.

The [isolated step-9 replay](../evidence/smolvla_reference_step_replay_20261014/README.md)
uses the exact full-run compiler flow object with captured FP32 state and
prefix cache. Its output has 0.907 cosine to the FP32 reference, compared
with 0.054 for step 9 in the full compiled session. This resets both the
prefix cache and previous flow state, so their contributions are not yet
separated. The isolated run has a different cold-start timing boundary and
no valid cycle ratio. All ten full-run timestep hashes follow BF16-rounded
VPU addition despite an f32 ABI; that is a measured precision difference,
not a quantified explanation of the entire action error.
The [scalar-state control](../evidence/smolvla_scalar_state_precision_20261014/README.md)
removes only the one-element FP32 VPU route. Its isolated Spike output
matches the old flow output byte for byte, while its timestep matches the
captured FP32 update exactly. The completed corrected full session confirms
that the timestep remains exact through all ten recurrent steps.
The [prefix-cache control](../evidence/smolvla_prefix_factor_20261014/README.md)
holds compiled step-8 flow state and timestep fixed while replacing only
the prefix cache with the captured FP32 value. Step-9 cosine to FP32 moves
from 0.054 to 0.066, compared with 0.907 when both cache and prior flow
state are reset. This narrows the immediate step-9 difference to accumulated
flow state; it does not isolate how earlier prefix errors affected that state.

Read the first two rows as **MX matrix command code comparisons**. The
compiler and Nicolas variants run the same Merlin host program and VPU code;
only the MX issuers change. The small full-model cycle gaps are therefore
expected and cannot establish parity with an independently scheduled model.
The deck generator checks the changed-object receipts and refuses to label
a comparison as an issuer swap if any other stage object changed.
The third row reports the **independently written 160-token TinyLlama
forward** through all 22 layers and the LM head. Its completed source
counter is 12,703,713,784 modeled cycles; the matched compiler counter is
15,170,996,572. Both programs use all 200 FP8 MX matrix sites and matched IDs
and checkpoint tensors, but their host/VPU placement and timer boundaries
differ. They select the same top token at 24/33 recorded late positions.
The separate
[128-token prefill control](../evidence/nicolas_tinyllama_prefill128_source_20261014/README.md)
records 9,611,186,417 source and 11,442,665,332 compiler cycles with
0.9454 logit cosine; it is no longer the slide row. The last slide row
compares captured SmolVLA final-step attention heads at 113 and 163 keys.
No row establishes an independent complete-session speedup. Spike cycles
are simulator results, not FPGA latency.

The [equal-work attention receipt](../evidence/attn_vpu_equal_work_20261013/README.md)
records 43,523 compiler versus 43,641 Nicolas source Spike cycles for a
64×256×128 dense MX+VPU attention kernel. Both runs pass the same output
checks. The [SmolVLA issuer substitution
receipt](../evidence/nicolas_matrix_substitution_smolvla_20261013/README.md)
records 2,112,653,481 compiler versus 2,112,919,925 Nicolas native issuer
cycles for one flow step, with identical declared output hashes.

The [full SmolVLA comparison](../evidence/nicolas_matrix_substitution_smolvla_full_20261013/README.md)
records 72,066,544,354 compiler cycles and 71,875,683,634 Nicolas native
matrix-issuer cycles over all 12 steps. All 33 output digests match. The
[full TinyLlama comparison](../evidence/nicolas_matrix_substitution_tinyllama_full_20261014/README.md)
records 83,547,564,353 compiler cycles and 83,544,602,795 Nicolas issuer
cycles over one prefill and 32 decode steps. All 99 output digests match.
Both full-model experiments change MX matrix issuers while holding the host
graph and VPU objects fixed.

The deck also contains a separate [Radiance MX kernel
control](../evidence/radiance_kernel_comparison_20261014/README.md). Its
source-bound compiler commands match all 16,384 BF16 outputs of each FP8
256×64×2048 and FP4 256×64×5632 read-once GEMM, with exactly 128 KiB and
176 KiB of weight reads. The equal-shape Radiance FP8 four-pass re-stream
control reads 512 KiB. These validate output and weight traffic; the original
Radiance kernels issue MX commands through Muon and have no paired cycles on
Nicolas's Rocket Spike model. Radiance's published flash-attention cycles
include Muon SIMT work and are not compared to the MX+VPU Spike cycles. The
separate source-produced GQA PV control matches 65,536 BF16 values on Spike;
it does not compile the whole mixed Muon+MX kernel.

The [independent full-depth VPU source
control](../evidence/nicolas_tinyllama_full_source_vpu_20261014/README.md)
reports **4,328,997,732** modeled source cycles and **5,072,967,821**
modeled compiler cycles. Its source logits are bit-identical to the earlier
scalar-source run; the compiled logits have 0.9816 cosine to source and the
same top token. The compiler harness reads the extension's modeled `cycle`
CSR. Older [source](../evidence/nicolas_tinyllama_full_source_20261014/README.md)
and [compiler](../evidence/tinyllama_full_safe_vpu_20261014/README.md)
64-token controls used default `func` mode and are separate numerical
diagnostics, not the slide's modeled row.

An [independent two-layer TinyLlama source smoke
test](../evidence/nicolas_tinyllama_two_layer_source_20261014/README.md)
passes every generated stage check on the same Spike extension. Its separate
timers report 6.20 million mesh cycles and 347.48 million scalar host cycles.
It uses 64 tokens, two FP8 layers, and host scalar operations, so those
counters are not a cycle baseline for the compiler's captured full session.

A handwritten fused MX+VPU baseline for the complete captured 33-step
TinyLlama or 12-step SmolVLA session has not been measured. For example,
Nicolas's `attn_vpu_fa.c` wrapper fixes a 64×256×128 dense attention shape;
the captured SmolVLA flow contains 15 heads of 50 queries against 163 keys,
with a boolean key mask. A matched fused replacement must preserve that
padding and mask behavior before its cycles can be compared.
The independent cached-layer probe adapts Nicolas's FP8 source to 64 cached
plus 64 query tokens. The captured full-model session has a 128-token prefill
and one-token decode with 155 FP4 and 45 FP8 MX bindings per stage. A stitched
source baseline needs those shapes and precision choices to match before its
cycle count can stand beside the model result.
The [matched source baseline requirements](../stitched_model_source_baseline.md)
record the remaining model and kernel interfaces.
The [cached TinyLlama layer probe](../evidence/tinyllama_cached_layer_compiler_20261014/README.md)
now runs a real 64-query layer from PyTorch through model2MLIR and the MX
compiler. It exposes per-operation packing and intermediate conversion costs,
but its stage cycle window differs from Nicolas's timed source kernel.
The cached-layer measurements remain in that evidence directory; the slide
now uses the full source run because it covers all decoder layers and the head.

## Slide 4: Radiance kernel comparison

The fourth slide gives the Radiance controls their own table. It checks
compiler BF16 outputs against the source goldens and compares physical tiles
and weight reads with the pinned Radiance kernels. For the 256×64×2048 FP8
case, the compiler uses one 256×64 output tile and reads 128 KiB of weights
once. The Radiance four-pass control uses four 64×64 tiles and reads 512 KiB.
The FP4 256×64×5632 down-projection reads 176 KiB. The slide also identifies
four batched GEMV controls and the narrower GQA PV cutpoint. Radiance source
uses a Muon+MX system; these compiler controls use Nicolas's standalone
Rocket+MX+VPU Spike model. The output and schedule checks do not establish
a cross-system cycle ratio.

These are source-shaped controls. The [FP8 payload-bound
MLIR](../evidence/radiance_ws_read_once_1a7bc9c/gemm_mxgemmini_ws/payload_bound.mlir)
records `payload_origin = "radiance_source_header_specialization"`: packed
codes and scales are bound from a Radiance header specialization, and the
tile policy is selected to match the source schedule. The comparison proves
that those inputs lower to matching outputs and physical traffic. It does
not establish automatic schedule discovery for arbitrary model operators or
complete mixed Muon+MX kernel compilation.

## Regenerate

Install `python-pptx`, then run:

```sh
python tools/build_mx_overview_deck.py \
  --template /path/to/Lab_Pastel_Template.pptx \
  --smol-full-comparison \
    docs/evidence/nicolas_matrix_substitution_smolvla_all_fp8_20261014/comparison.json \
  --smol-full-mode all-FP8 \
  --tiny-full-run \
    docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/compiler/mx_session_run.json \
  --tiny-full-comparison \
    docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/comparison.json \
  --tiny-layer-probe \
    docs/evidence/tinyllama_cached_layer_compiler_20261014/summary.json \
  --tiny-full-source \
    docs/evidence/nicolas_tinyllama_full_source_20261014/receipt.json \
  --tiny-full-safe-vpu \
    docs/evidence/tinyllama_full_safe_vpu_20261014/receipt.json \
  --tiny-full-vpu \
    docs/evidence/nicolas_tinyllama_full_source_vpu_20261014/full_receipt.json \
  --radiance-comparison \
    docs/evidence/radiance_kernel_comparison_20261014/receipt.json \
  --equal-work-attention \
    docs/evidence/attn_vpu_equal_work_20261013/receipt.json \
  --smol-attention-pair \
    docs/evidence/smolvla_attention_session_20261014/compiled_pair_pipeline/receipt.json \
  --tiny-prefill128 \
    docs/evidence/nicolas_tinyllama_prefill128_source_20261014/receipt.json \
  --tiny-prefill-precision \
    docs/evidence/tinyllama_prefill_precision_20261014/receipt.json \
  --tiny-first11-prepacked \
    docs/evidence/tinyllama_prefill_precision_20261014/first11_prepacked_spike_receipt.json \
  --tiny-full-logits \
    docs/evidence/tinyllama_full_logits_20261014/receipt.json \
  --smol-action-values \
    docs/evidence/smolvla_full_action_values_20261014/receipt.json \
  --smol-reference-step \
    docs/evidence/smolvla_reference_step_replay_20261014/receipt.json \
  --smol-prefix-factor \
    docs/evidence/smolvla_prefix_factor_20261014/receipt.json \
  --smol-corrected-action \
    docs/evidence/smolvla_scalar_full_action_20261014/receipt.json \
  --smol-all-fp8-action \
    docs/evidence/smolvla_all_fp8_full_action_20261014/receipt.json \
  --tiny-all-fp8-session \
    docs/evidence/tinyllama_all_fp8_full_session_20261014/against_fp32.json \
  --tiny-onepass-source \
    docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014/source_timed_summary.json \
  --tiny-onepass-comparison \
    docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014/matched_onepass_comparison.json \
  --out docs/presentations/mx_gemmini_dialect_overview.pptx
```

The script keeps the template's architecture, process, and results layouts,
adds a Radiance results slide with native PowerPoint objects, and labels
each comparison. The generator accepts the full TinyLlama run/comparison
receipts only when the complete 33-step run and equal-output comparator succeed.
Its optional `--tiny-prefill128` row requires the archived source and compiler
Spike logs, their hashes, and the verified modeled cycle counters. A preview
receipt or an incomplete run cannot replace the measured row.

The results callout reports TinyLlama logit drift and the mixed versus
all-FP8 SmolVLA final-action comparison to captured FP32 on fixed synthetic
inputs. The generator checks archive hashes and binds each full-session
receipt to its timed run. These metrics are diagnostics rather than
generation or robotics task-success thresholds.
