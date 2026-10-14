# MX Gemmini dialect overview slides

[Editable PowerPoint deck](mx_gemmini_dialect_overview.pptx) contains three
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
correspond to `resource` + `contract` + readout; its vector postprocessing
corresponds to `vpu_execute`, `spad_requant`, and `wait`. The slide shows a
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

Read the first three rows as **MX matrix command code comparisons**. The
compiler and Nicolas variants run the same Merlin host program and VPU code;
only the MX issuers change. The small full-model cycle gaps are therefore
expected and cannot establish parity with an independently scheduled model.
The deck generator checks the changed-object receipts and refuses to label
a comparison as an issuer swap if any other stage object changed.
The last row compares compiler-issued commands
with one fused source attention kernel at the same 64×256×128 shape. These
timings do **not** compare the compiler with Nicolas's fused kernels stitched
into a full TinyLlama or SmolVLA model. Matching output hashes prove agreement
with the captured
outputs; they do not measure model accuracy against an independent reference.
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

A whole-model handwritten fused MX+VPU baseline on the same captured graphs
has not been measured, so the slide does not estimate its cycles. For example,
Nicolas's `attn_vpu_fa.c` wrapper fixes a 64×256×128 dense attention shape;
the captured SmolVLA flow contains 15 heads of 50 queries against 163 keys,
with a boolean key mask. A matched fused replacement must preserve that
padding and mask behavior before its cycles can be compared.
[Nicolas's native full-layer TinyLlama kernel][nicolas-layer]
measures an `M=32` FP8 schedule. The captured session has a 128-token prefill
and one-token decode with 155 FP4 and 45 FP8 MX bindings per stage. A stitched
source baseline needs those shapes and precision choices to match before its
cycle count can stand beside the model result.
The [matched source baseline requirements](../stitched_model_source_baseline.md)
record the remaining model and kernel interfaces.
The [cached TinyLlama layer probe](../evidence/tinyllama_cached_layer_compiler_20261014/README.md)
now runs a real 64-query layer from PyTorch through model2MLIR and the MX
compiler. It exposes per-operation packing and intermediate conversion costs,
but its stage cycle window differs from Nicolas's timed source kernel.
Those two numbers are kept off the slide's paired table.

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
  --out docs/presentations/mx_gemmini_dialect_overview.pptx
```

The script keeps only the template's architecture, process, and results
layouts, edits their native objects, and adds source-specific labels and
figures. The generator accepts the full TinyLlama run/comparison receipts only
when the complete 33-step run and equal-output comparator succeed.

[nicolas-layer]: https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/llama_layer_full_native.c
