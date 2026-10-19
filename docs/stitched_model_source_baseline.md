# Full-model source baselines

The canonical [comparison guide](fair_full_model_comparisons.md) distinguishes
three experiments that require different interpretations:

1. **Independent program:** TinyLlama's complete 160-token forward runs the
   adapted `llama_model.c` from pinned `npu-exploration` sources, including
   embedding lookup, all 22 layers, and the output head. It has matched input
   and checkpoint data and a separately checked source-relative numerical gate.
2. **Issuer substitution:** historical complete-session controls keep Merlin's
   host graph, VPU objects, state adapter, and outputs fixed while replacing
   the MX matrix-issuer catalog. This measures issuer overhead within the
   compiler's schedule.
3. **Radiance source controls:** pinned `radiance-kernels` GEMM/GEMV drivers
   supply packed operands, BF16 references, and tile/traffic schedules. These
   do not supply an independently stitched complete MX+VPU model application
   or paired standalone Rocket-Spike cycle measurements.

The word “stitched” in older artifact identifiers refers to the second
experiment. It must not be used to imply the third repository supplied a
complete independent model. Stable historical paths are retained so that
published artifact hashes and links remain usable.

A new independent cached TinyLlama or complete SmolVLA baseline must execute
the same captured inputs, checkpoint, masks, recurrent state, stage schedule,
and logical output boundary. Use the same target profile, simulator binary,
extension, timing settings, and preparation accounting. Publish all output
and state comparisons, a numerical acceptance criterion, and the actual
source adaptation before calculating a full-model speedup.

See the [compilation guide](compile_e2e_models.md) for model and tool versions,
and the [audit](model_results_audit.md) for the checks performed on the current
results.
