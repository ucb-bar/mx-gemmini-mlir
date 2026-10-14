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
8-lane BF16 VPUs, and `SPAD_REQUANT`. FP6 is available in a separate
non-VPU configuration. Nicolas's `npu-exploration` kernels provide source
checks and performance reference programs, while RTL defines hardware
capabilities. See [profile inventory](../current_mx_profiles.md).

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

The captured TinyLlama prefill and decode stage catalogs each have 200 MX
`bindings`: 155 FP4 and 45 FP8, counted from each binding's
`target_dtypes`. The prepacked catalog has 158 distinct `kernels` because
some bindings share an issuer; kernel count is not operation count. The
[full-model compilation guide](../compile_e2e_models.md) describes how to
regenerate the stage catalogs and inspect their precision choices.

## Slide 3: measured scope

The rows answer different questions. They must not be read as one
whole-model compiler-versus-handwritten speedup:

| Row | Changed work | What the result tests |
| --- | --- | --- |
| Full TinyLlama and SmolVLA sessions | MX matrix issuers only; host graph and VPU objects stay fixed | MX command emission and its cost inside the captured program |
| TinyLlama independent prefill | Complete source program versus compiled prefill, with different host and VPU work | Functional agreement and a diagnostic raw cycle gap for one prompt |
| Dense attention | One matched 64×256×128 fused kernel | Kernel-level code and modeled timing at that shape |

The requested full-model performance comparison needs an independently
stitched MX+VPU program for each captured session, matching inputs, weights,
precision, shapes, outputs, and Spike settings. Neither full-session
issuer swap supplies that program. The independent prefill is only one
64-token, all-FP8 stage; the captured full sessions use mixed FP4/FP8.
Its raw cycle counts do not supply a full-model speedup.

Read the first two rows as **MX matrix command code comparisons**. The
compiler and Nicolas variants run the same Merlin host program and VPU code;
only the MX issuers change. The small full-model cycle gaps are therefore
expected and cannot establish parity with an independently scheduled model.
The deck generator checks the changed-object receipts and refuses to label
a comparison as an issuer swap if any other stage object changed.
The third row compares an **independently written full prefill program**
against the compiled 64-token, 22-layer TinyLlama stage. Both use the modeled
`e4m3_vpu` clock, the same 200 FP32 checkpoint tensors, and identical input
tokens. Their logits have 0.9816 cosine and the same top token. The source
uses 44 VPU residual-add calls while the compiler lowers 88 add regions and
uses different host math. The compiled attention has 65 allocated key slots
where the source computes 64 valid keys. The slide therefore prints raw
counts with **NO RATIO**.
The last row compares one fused source attention kernel at the same
64×256×128 shape. None of these rows compares a fused source implementation
of a complete TinyLlama or SmolVLA model. The issuer-swap output hashes show
agreement between those two executables, not independent model accuracy.
Spike cycle counts are simulator results, not FPGA latency.

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

The slide also names the separate [Radiance MX kernel
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
    docs/evidence/nicolas_matrix_substitution_smolvla_full_20261013/comparison.json \
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
  --out docs/presentations/mx_gemmini_dialect_overview.pptx
```

The script keeps the template's architecture, process, and results layouts,
adds a Radiance results slide with native PowerPoint objects, and labels
each comparison. The generator accepts the full TinyLlama run/comparison
receipts only when the complete 33-step run and equal-output comparator succeed.
Its optional `--tiny-prefill128` row requires the archived source and compiler
Spike logs, their hashes, and the verified modeled cycle counters. A preview
receipt or an incomplete run cannot replace the measured row.
