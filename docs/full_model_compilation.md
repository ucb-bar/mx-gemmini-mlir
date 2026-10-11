# Full model compilation through model2MLIR and MX Gemmini

The [four-model precision proposals](evidence/model_precision_proposals_a042643_266c593/README.md)
bind the current FP8/FP4 placement intent to each archived model2MLIR graph
and the pinned MX+VPU profile. They are placement plans, not executable
whole-model programs. FP8 is the general baseline and the active attention
QK/PV choice; aligned ordinary matmuls are proposed for FP4.

The [FP4 projection archive](evidence/model2mlir_four_model_fp4_vpu_a042643_266c593/README.md)
records one full-width `q_proj` matmul from each reduced random-weight model
executed on pinned MX Spike, with every BF16 output matched to an independent
MX arithmetic reference. Qwen and SmolVLA bias additions and all other model
operators remain outside those single-matmul programs. Full-model execution
still needs partitioning, buffer routing, other operator lowerers, and session
state.

The acceptance target is a complete inference session, from model2MLIR capture
through a standalone target executable. For the causal language models this
means prefill, repeated decode, KV-cache updates, and logits with the intended
checkpoint weights. For SmolVLA it means image/language/state prefix encoding,
every flow step, and action decoding. Target execution must need no PyTorch or
Python fallback, and outputs must be checked against the same model and inputs
in PyTorch. A graph capture or an MX matrix object alone does not pass this
gate.

The default [Merlin target contract](../contracts/target_contract.yaml) is now
generated from the pinned `MxE4M3Fp4VpuGemminiRocketConfig` profile. It
declares the direct FP4 E2M1 and FP8 E4M3 PE modes, BF16 accumulation, and
the source-observed BF16 row softmax VPU capability. It remains marked
`structural_unqualified` and `phase0_admitted: false`: a mode declaration is
not an executable catalog or a numerical result. The previous standalone
prototype contract is retained as
[`target_contract-f016-historical.yaml`](../contracts/target_contract-f016-historical.yaml).
Regenerate and check the selected contract with:

```sh
python -m mx_gemmini_support.merlin_profile_contract \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out contracts/target_contract.yaml --check
```

For a current full-checkpoint session bundle, run:

```sh
python -m tools.audit_model2mlir_session --session /path/to/session \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out /path/to/new-audit.json
```

The tool verifies bundle integrity, inventories each stage through Merlin,
and reports FP4/FP8 candidate counts alongside all
remaining operation families. Its `unlowered_operation_count` intentionally
equals the stage's operation count until an executable catalog and host/device
partition have been built and verified.

On 2026-10-11, the current model2MLIR `02c61c2` plus the changes on this
branch captured all four full pretrained checkpoints as explicit sessions.
Every stage had zero opaque operations and passed the model2MLIR stage-receipt
and whole-bundle integrity checks. The MX structural audit used Merlin's shared
parser and the selected Nicolas MX+VPU profile. Counts below sum distinct
stage graphs; repeated decode or flow steps execute those graphs again.

| Session | Stage graphs | MLIR operations | Proposed FP4 matmuls | Proposed FP8 QK/PV | Other contraction regions |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama | 2 | 7,245 | 310 | 88 | 2 |
| DeepSeek Qwen 1.5B | 2 | 8,851 | 394 | 112 | 170 |
| Gemma 2 2B | 2 | 10,107 | 366 | 104 | 2 |
| SmolVLA | 3 | 9,564 | 302 | 25 | 141 |

Qwen's other regions are predominantly the 84 bias epilogues in each stage;
their matrix components are separately proposed for FP4. SmolVLA has
additional contraction shapes and ABI forms that the current MX importer
retains for host or further VPU legalization. These are candidate counts,
with no executable MX placement, quantized reference, linked session, or Spike
inference result. The captures use synthetic inputs and are not paper-ready
quality measurements. SmolVLA was recaptured with xDSL 0.65 so all four graphs
can be parsed in the same Merlin environment.

## First compiler probe

The [four-model preflight archive](evidence/full_model_compile_preflight_a042643_748b984/README.md)
uses committed model2MLIR `a042643` and this repo's Nicolas Rocket profile.
All four reduced, random-weight captures have zero opaque frontend operations.
They are structural probes, not complete checkpoints or sessions. The current
`tools.compile_object` entry point declines all four raw model2MLIR modules.
It parses only verified MX graphs from a small set of qualified executable
families and has no general whole-model lowering, CPU lane, or model runtime.

| Probe | Captured scope | Matrix ops | MX object |
|---|---|---:|---|
| TinyLlama | 2/22 layers, tokens to logits | 15 | Declined |
| DeepSeek-R1-Distill-Qwen-1.5B | 1/28 layers, tokens to logits | 8 | Declined |
| Gemma 2 2B | 2/26 layers, tokens to logits | 15 | Declined |
| SmolVLA | 1 VLM + 1 expert layer, one denoise step | 93 | Declined |

Gemma's gated checkpoint configuration was unavailable offline during this
probe, so its capture uses the installed `Gemma2Config` defaults with random
weights. SmolVLA uses LeRobot 0.5.1 in a task-local environment with random
weights and a 64×64 image. Neither is a pretrained-output parity test.

The MX [whole-graph contraction importer](../mx_gemmini_support/model2mlir_worklist.py)
now strictly parses and verifies all four captured modules. It preserves
source region IDs, module names, rank-two shapes, element types, and the other
contraction families in a target-profile-bound worklist. The four probes
contain respectively **20, 14, 20, and 198** contraction regions; only
**15, 8, 15, and 93** currently appear as rank-two `linalg.matmul` operations.
SmolVLA has 14 BF16 rank-two matmuls and seven whose K dimension needs MX
32-element scale-group padding. The remaining groups include batched
attention, bias-bearing `addmm`, and an im2col convolution. The worklist is
an input to future lowering and carries no executable placement claim.

Produce one worklist with:

```sh
python -m tools.plan_model2mlir_worklist \
  --mlir docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz \
  --reader merlin \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --out /new/tinyllama-mx-worklist.json
```

## Integration path

1. Have Merlin ingest the complete model2MLIR module, parameter manifest, and
   persistent-state ABI. Merlin remains target agnostic: it owns graph
   semantics, partitions, lifetimes, and the whole-program build contract.
2. Select RTL-legal MX contractions and VPU operations with a target recipe.
   The MX dialect owns their precision, scale/LUT binding, transfers,
   scratchpad allocation, command ordering, and RoCC emission. Define a
   compiled RV64 host lane for embedding, normalization, rotary position,
   masks, softmax, control, and any operation without a legal MX mapping.
   Extend the worklist importer to lower the rank-two, batched, and bias-bearing
   sites while preserving their original SSA dependencies.
3. Link both lanes with explicit buffers and synchronization into one
   standalone executable. Verify that every captured op has an executable
   owner and that no runtime input or checkpoint weight is silently omitted.
4. Prove a small complete TinyLlama prefill+decode session first. Then extend
   to Qwen's biased projections and grouped-query attention, Gemma's sliding
   attention and soft-capping, and SmolVLA's vision prefix and recurrent action
   session. Move from reduced random-weight probes to full-depth pinned
   checkpoints and compare every output and recurrent state against PyTorch.

The existing six connected FP8/FP4/FP6 source chains validate the MX object
lowerer for those graph families. They do not establish any of the complete
model gates above.

## Reuse Merlin's whole-model path

Merlin already owns model2MLIR capture bundles, external checkpoint weights,
host lowering for the rest of a graph, source-region accounting, an explicit
device catalog hook, bare-metal linking, Spike execution, and multi-program
session bindings. The MX compiler should implement a **target-owned catalog
builder** for Merlin's `DeviceRouting.catalog_builder` boundary. It should
not create another model graph compiler or host runtime here. The latest
model2MLIR `main` checked on 2026-10-11 was `02c61c2`; existing archived MX
captures remain pinned to their older recorded revisions until recaptured.

The first executable handoff is `tools.emit_runtime_issuer`. It takes a
payload-bound model2MLIR-derived MX program and emits a data-free RV64 object
with six pointer arguments: packed activation codes, activation E8M0 scales,
packed weight codes, weight E8M0 scales, BF16 output, and scale scratch. It
binds the exact source payload and target profile, records the required buffer extents,
and rejects a non-row-major or non-BF16 schedule. The object has no embedded
operand, golden, or allocated data sections. A two-call TinyLlama projection
diagnostic [passed on Nicolas's pinned Spike](evidence/runtime_issuer_spike_tinyllama_266c593/README.md)
with the same tensor bytes at two
distinct address sets: both 512-element BF16 outputs matched the independent
archived golden. This establishes a live **encoded** operand ABI for one
projection, not a Merlin dense-tensor call or whole-model inference.

To build that object from the checked-in projection:

```sh
python -m tools.emit_runtime_issuer \
  --mlir docs/evidence/model2mlir_tinyllama_projection_a042643_266c593/projection.payload_bound.mlir \
  --bundle docs/evidence/model2mlir_tinyllama_projection_a042643_266c593/bundle \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/pinned/riscv-tools \
  --out-dir /new/output/directory
```

The remaining target work is an MX catalog entry that accepts Merlin's live
dense tensors, performs exact FP8/FP4 quantization and padding, invokes this
issuer, converts BF16 readout back to the graph's type, and records one source
region per call. Batched attention and bias-bearing contractions require their
own legalizations. Merlin's existing host path should run all remaining graph
operations and its session runtime should carry KV caches and SmolVLA action
state. The 4096-element output limit in Merlin's saved-model diagnostic also
needs a larger or chunked readback for complete LM logits; it is an execution
tooling limit, not a reason to write another graph runtime. Full checkpoint
captures and reference outputs must be selected for each named model before
claiming pretrained inference.

The MX runtime now has a freestanding C encoder for live f32 operands. It
reproduces the pinned MXQuant BF16/E8M0 FP8 and FP4 bytes, including the
target's different activation and weight packing orders. A [pinned Spike
replay](evidence/runtime_f32_projection_spike_266c593/README.md) pads two
small-LLaMA FP4 MLP projections at runtime, converts their live f32 operands,
calls the data-free issuer, and checks all 15,360 BF16 readouts. These are
separate projections; the prepared-model catalog and complete model link
remain open.

`mx_gemmini_support.dense_kernel.emit_dense_f32_kernel` now emits the target
side of Merlin's dense `lhs, rhs, out` f32 ABI for one already qualified MX
schedule. It pads the live operands, packs FP4/FP8 codes and E8M0 scales,
calls the six-pointer RoCC issuer, and copies the logical region of its BF16
readout back to f32. Its buffers have static storage, so this first adapter
is serial and non-reentrant. The compiler must account for its workspace
against the selected memory map before admitting a full model. The
`--dense-adapter` mode of `tools.qualify_runtime_f32_projection_spike` checks
the generated entry on Nicolas's pinned Spike against the archived projection
reference. On the gate and down sites, respectively, all 2,752 and 1,024
logical f32 outputs matched the archived BF16 reference. Each run also
combined the adapter, issuer, and quantizer into a relocatable RV64 object
without unresolved symbols. This is the per-kernel ABI handoff. A
prepared-model catalog builder, exact source-operation bindings, and session
execution remain open.

The regular matrix command order is now shared by source qualification and
`emit_planned_runtime_issuer`. The latter takes the selected profile, FP4/FP8
mode, padded dimensions, and tile without any operand or golden payload.
`plan_dense_shape` chooses a conservative mesh tile from a logical M×N×K
shape, and `emit_dense_contraction` combines that plan with the dense adapter.
For both archived small-LLaMA projections, the shape-planned issuer emitted
the exact same C command stream as the payload-qualified path; the gate site
also ran through the shape-planned entry on Nicolas's Spike with all 2,752
logical outputs matching. This establishes command parity for those shapes.
It does not establish a general numerical qualification or full-model
placement for the other shapes in the captured sessions.

Large model dimensions use `emit_compact_runtime_issuer`, which emits C loops
over output tiles and K waves. A host trace compares every command operand and
fence against the source-qualified unrolled schedule for FP4 gate/down and
an FP8 TinyLlama projection. Both FP4 sites also passed on Nicolas's pinned
Spike with the compact issuer: 2,752 and 1,024 logical output values matched.
The compact source stays under 7 KiB even for an 18,992-tile vocabulary
projection. These tests qualify command order and those executed shapes;
they do not establish numerical accuracy for every model site.

`mx_gemmini_support.merlin_catalog_plan` now binds selected FP4 rank-two and
FP8 attention contractions to exact Merlin source regions, operation
ordinals, shapes, and tensor types. `merlin_catalog_builder` emits one
relocatable RV64 object and a hashed source-conversion manifest for Merlin's
`DeviceRouting.catalog_builder` callback. Equal shapes share a kernel, and
all kernels reuse one aligned serial workspace. On the archived two-layer
small-LLaMA graph, 19 operations mapped to six kernels and Merlin's rewrite
and catalog binder accepted the object and shim. On full checkpoint TinyLlama
prefill, 199 selected operations mapped to seven kernels, with a 43,149,312
byte shared workspace; the full graph rewrite and binder accepted all 199
routes without skipped calls. The catalog status remains
`compiled_experimental_numeric_qualification_pending`.

Exact source binding now inventories all four full-checkpoint sessions. The
current f32 catalog binds 199 sites in each TinyLlama stage, 253 in each Qwen
stage, and 235 in each Gemma stage. SmolVLA binds 139 prefix-encode and 116
flow-denoise sites. The 72 remaining SmolVLA prefix proposals are BF16
`addmm` matrix components: they require a BF16 source ABI and target-owned
conversion, which the current uniform-f32 catalog does not implement.
Action decode contains no direct MX contraction proposal. The planner reports
these 72 regions explicitly as unbound; the catalog's `coverage_complete`
only means that every operation listed in its exact binding table has a
compiled implementation. VPU operators and other host operations are not
covered by that field.

For a prepared graph, the standalone builder is:

```sh
MERLIN_TARGET_PATH=/path/to/mx-gemmini-mlir \
python -m tools.build_merlin_catalog \
  --prepared /path/to/model.prepared.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --out-dir /new/catalog-output
```

The remaining full-model gates are Merlin's host graph object and final ELF,
serial workspace admission against a selected memory map, full session
link/run on Nicolas's Spike, VPU placement beyond the existing softmax
contract, and comparison of every output and recurrent state against an
independent quantized PyTorch reference. The native f32 encoder and packed
buffer glue stay in this target package; Merlin's graph inventory, catalog
ABI, memref shim, host lowering, and session orchestration remain shared.

## Compiler ownership

The `mx_gemmini_support` package is the MX backend's Python and C support,
not a replacement for Merlin's graph compiler. Merlin already owns the
target-neutral `linalg-on-tensors` structural reader, model and session
orchestration, source-bound catalog binding, dense memref shim, host lowering,
and link/runtime infrastructure. The MX package owns only MX profile legality,
precision and E8M0 policy, packed physical layout, command generation, and
target ABI glue. `--reader merlin` now projects Merlin's shared structural
inventory into the MX worklist; on the four archived model graphs it produces
identical MX precision plans. The older local reader remains for standalone
kernel evidence, but it must not become a second whole-model graph frontend.
The catalog builder belongs in this target package, but it must implement
Merlin's generic `DeviceRouting.catalog_builder` interface.

Merlin's catalog route now has an explicit, target-neutral source dtype
conversion contract on `main` at `6f1acb6e6`. That admits
an f32 model contraction only when the catalog names a real target-native
datapath and a hashed numeric policy. The selected Nicolas profile now exports
direct FP4 and FP8 datapath facts into Merlin, satisfying that structural gate.
The conversion route still requires a target-owned dense f32 catalog wrapper,
source-bound numeric policy, and a runnable object. This is not a reason to
label f32 as an MX hardware dtype. `runtime_quant.c` also still mixes reusable
numerical encoding with MX physical packing; the generic numerical part should be
promoted to Merlin when its shared C runtime API is established, while the
target's byte layout and command ABI stay here.

The [current model2MLIR small LLaMA probe](evidence/model2mlir_small_llama_precision_e9ded36_266c593/README.md)
uses revision `e9ded36` and captures a complete two-layer random-weight
LLaMA-style graph with zero opaque operations. Its precision proposal now
recognizes both `attn` QK/PV pairs as FP8 sites and records the padded MX
shapes and FP4 placement for the six unaligned MLP matmuls. It remains a frontend and
placement probe, with no whole-model executable.

The [first padded FP4 MLP archive](evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593/README.md)
and [remaining-site archive](evidence/model2mlir_small_llama_remaining_mlp_fp4_e9ded36_266c593/README.md)
execute all six formerly unaligned gate, up, and down projections in both
layers. Source dimension 344 in N or K is zero-padded to 352. Pinned Spike
matched all 53,248 padded BF16 outputs against the independent MX reference,
including 13,056 logical model outputs. Each is a separate program; the
nonlinear MLP path and whole-model graph remain to be connected.

## First executed model-derived contraction

The [TinyLlama projection archive](evidence/model2mlir_tinyllama_projection_a042643_266c593/README.md)
now compiles a full-K, 32-column stripe of layer-0 `q_proj` from the captured
model site. A live two-layer TinyLlama forward supplies actual f32 activation
and weight tensors; MXQuant produces FP8 codes and E8M0 scales. The typed MX
contraction lowers to a data-free RV64 object and a standalone ELF. Pinned
Spike matches an independent hardware arithmetic reference on all 512 BF16
outputs; 256 belong to the eight real token rows and 256 to required DIM16
padding. The module remains only one stripe of one projection. The full
TinyLlama, Qwen, Gemma, and SmolVLA compile gates above remain open.

The same path now covers the [complete layer-0 TinyLlama q_proj](evidence/model2mlir_tinyllama_full_qproj_a042643_266c593/README.md):
8×2048×2048 before row padding, all 2048 output columns, and all K groups.
Its one compiled object matched 16,384 model outputs and 16,384 padded outputs
on pinned Spike. This advances matrix lowering scale while the whole-program
acceptance gate remains open.
