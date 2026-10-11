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
separate projections; the dense three-pointer catalog wrapper and complete
model link remain open.

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
conversion contract on its `cx/mx-catalog-source-abi` task branch. That admits
an f32 model contraction only when the catalog names a real target-native
datapath and a hashed numeric policy. The current MX provider contract still
declares no executable compute-unit datapaths, so the conversion route cannot
be enabled for MX until the selected Nicolas profile exports those facts into
Merlin. This is target profile integration work, not a reason to label f32 as
an MX hardware dtype. `runtime_quant.c` also still mixes reusable numerical
encoding with MX physical packing; the generic numerical part should be
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
