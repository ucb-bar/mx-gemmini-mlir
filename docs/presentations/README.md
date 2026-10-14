# MX Gemmini dialect overview slides

[Editable PowerPoint deck](mx_gemmini_dialect_overview.pptx) contains three
slides based on the supplied Lab Pastel PowerPoint template. The diagrams,
text, callout, and results table are PowerPoint objects. The university seal
is the template's master image.

## Slide 1: compiler structure

The C++ dialect defines typed MX resource, contraction, readout, VPU,
requantization, and ordering operations in
[`MxOps.td`](../../include/MxGemmini/MxOps.td). `mx-gemmini-opt` verifies this
IR. Profile and payload checks, tile planning, physical commands, and Merlin
integration currently live in Python modules under `mx_gemmini_support/`.
`MXBF16CatalogPreparation` and `MXVPUPostOffload` are Merlin-side
transformations, not registered C++ MLIR passes.

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

## Slide 3: measured scope

The [equal-work attention receipt](../evidence/attn_vpu_equal_work_20261013/README.md)
records 43,523 compiler versus 43,641 Nicolas source Spike cycles for a
64×256×128 dense MX+VPU attention kernel. Both runs pass the same output
checks. The [SmolVLA issuer substitution
receipt](../evidence/nicolas_matrix_substitution_smolvla_20261013/README.md)
records 2,112,653,481 compiler versus 2,112,919,925 Nicolas native issuer
cycles for one flow step, with identical declared output hashes.

The compiler's 12-step SmolVLA timing receipt records 72,066,544,354 modeled
cycles; full-session numeric qualification remains pending. The source-side
12-step issuer substitution is still running as of 2026-10-13. That
experiment changes MX matrix issuers while holding the host graph and VPU
objects fixed. A whole-model handwritten fused MX+VPU baseline on the same
captured SmolVLA graph has not been measured, so the slide does not estimate
its cycles.

## Regenerate

Install `python-pptx`, then run:

```sh
python tools/build_mx_overview_deck.py \
  --template /path/to/Lab_Pastel_Template.pptx \
  --out docs/presentations/mx_gemmini_dialect_overview.pptx
```

The script keeps only the template's architecture, process, and results
layouts, edits their native objects, and adds source-specific labels and
figures. Update the source-side 12-step cell only after the linked-session
comparison tool confirms matching output digests and timing receipts.
