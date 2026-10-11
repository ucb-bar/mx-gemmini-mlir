# Full model compilation through model2MLIR and MX Gemmini

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

## Integration path

1. Have Merlin ingest the complete model2MLIR module, parameter manifest, and
   persistent-state ABI. Merlin remains target agnostic: it owns graph
   semantics, partitions, lifetimes, and the whole-program build contract.
2. Select RTL-legal MX contractions and VPU operations with a target recipe.
   The MX dialect owns their precision, scale/LUT binding, transfers,
   scratchpad allocation, command ordering, and RoCC emission. Define a
   compiled RV64 host lane for embedding, normalization, rotary position,
   masks, softmax, control, and any operation without a legal MX mapping.
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
