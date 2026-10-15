# TinyLlama source and compiler: one 160-token input

This control gives Nicolas's handwritten MX+VPU source program and the
model2MLIR → Merlin → MX+VPU compiler program the **same 160 token IDs** and
the **same 200 FP32 checkpoint tensors**. The archived
[input identity receipt](input_identity.json) checks every tensor exactly.
The model2MLIR capture uses a prepared causal mask with the same eager
logits as the standard Hugging Face mask on an eight-token control. Its full
160-token FP32 forward has no opaque operations.
Run `python -B verify.py` to check the archived IDs, capture hashes, profile,
and FP8 recipe without the multi-gigabyte checkpoint files.

This is **one causal forward** through all 22 TinyLlama decoder layers and
the LM head. It is distinct from a 128-token prefill followed by 32 cached
decode calls. Both use 160 physical attention key positions, removing the
key-slot mismatch in the earlier prefill probe. Both use E4M3 MX operands
for all 200 matrix sites;
the compiler's [override map](all_fp8_overrides.json) selects them. The
compiler also offloads 22 softmax, 88 add, 358 elementwise, 156 unary, and
45 reduction regions to the VPU, as recorded in the
[compiler build summary](compiler_build_summary.json). Nicolas's source
keeps softmax, RoPE, normalization, and SwiGLU on the host and uses the VPU
for residual adds.

The source's timer begins with precomputed BF16 token embeddings; the
compiler's timer includes the token embedding lookup. These are the same
logical token inputs, but the measured cycle boundaries differ by that host
lookup and the operation placement above. A cycle ratio must carry those
qualifications. The source BF16 forward and the compiler FP32 capture have
33/33 matching top tokens at positions 127–159 before quantization; the
mean cosine of those rows is 0.999960. This verifies the input capture, not
the Spike program outputs.

The [independent timed source run](source_timed_summary.json) has now
completed on Nicolas's MX+VPU Spike: **12,703,713,784 modeled cycles** for
the 160-token forward. Its 33 emitted BF16 logit rows exactly match the
generated source reference. Against the captured FP32 logits, 27/33 late
top tokens match and mean cosine is 0.988564. The compressed
[Spike log](source_timed.log.gz) preserves the emitted rows and counter.
These source rows provide an independent check of the generated reference.

The [matched compiler run](matched_onepass_comparison.json) has now
completed at **15,170,996,572 modeled cycles**. Its 160 top-token IDs are
recorded in the [Spike log](compiler_spike.log). Nicolas's source and the
compiler select the same top token at **24/33** late positions. Against the
captured FP32 logits, the source matches **27/33** late top tokens and the
compiler matches **20/33**; the compiler matches **103/160** positions over
the entire forward. The compiler guest emitted the full-output digest and
top-token list, not its full logit tensor, so this comparison does not claim
a source-versus-compiler logit cosine. The source starts from precomputed
BF16 embeddings while the compiler includes token lookup, and their host/VPU
placements differ. The two cycle counts are diagnostic, not a speedup ratio.

Run `python -B verify_matched.py` with NumPy to check the archived IDs,
200 FP8 bindings, source BF16 rows, top tokens, receipts, and counters
without the multi-gigabyte checkpoint. The compact archive includes the
33 late captured FP32 rows and the generated source BF16 reference slice;
the [manifest](matched_archive_manifest.json) records each file hash and the
full source blob hash for reproduction.

## Reproduce the inputs

Use model2MLIR commit `902cb22` or later on `cx/mx-vpu-four-model-e2e`,
the pinned MX profile `MxE4M3Fp4VpuGemminiRocketConfig`, and Nicolas's
`npu-exploration` source at `0344bcba70b899039ccd0ac0801ca5605d46d3eb`.
The archived [input IDs](input_ids.npz) are the exact model2MLIR
`inputs.npz` file. Extract `in0` to a one-dimensional NumPy `.npy` file and
set `M2M_LLAMA_TOKEN_IDS` to that file with `M2M_SEQ=160` when capturing the
one-pass compiler program. The [model compilation guide](../../compile_e2e_models.md)
explains the model2MLIR bundle and stage build commands. Build the stage
with `--prepack-checkpoint-weights`,
`--precision-overrides all_fp8_overrides.json`, and
`--spike-arena-base 0xd0000000`.

To regenerate Nicolas's source input from the same IDs and checkpoint, use
`tools/capture_nicolas_tinyllama_onepass.py --input-ids input_ids.npz
--out-dir ACT_DIR`. Combine that activation capture with the checkpoint
weight capture using
`tools/generate_nicolas_tinyllama_onepass.py --weights-capture WEIGHT_DIR
--activation-capture ACT_DIR --source-generator NPU/baremetal/mxgemmini/gen/gen_llama_model.py
--out-dir SOURCE_DIR`. The generator needs the 22 `layerN.npz` weight files
and `model.npz` from Nicolas's pinned checkpoint capture, produced by
`python -m kernels.captures.llama_model` in that repository. Apply the earlier
[VPU residual patch](../nicolas_tinyllama_full_source_vpu_20261014/vpu_residual.patch),
the [128-token tiling patch](../nicolas_tinyllama_prefill128_source_20261014/m128_tiling.patch),
then this [160-token tile and logit patch](m160_tiling.patch) to Nicolas's
`llama_model.c` before compiling the source driver.

Check the two captures before timing:

```sh
python -m tools.verify_nicolas_tinyllama_onepass_input \
  --source-weights WEIGHT_DIR --source-activations ACT_DIR \
  --compiler-bundle COMPILER_BUNDLE --out input_identity.json
```

The compiler stage runner accepts `--spike-isa rv64gc_zicntr`,
`--perf-config e4m3_vpu`, and `--perf-set mem.host_tracking=0`, matching the
source Spike settings. After both guests complete, use
`tools/compare_nicolas_tinyllama_onepass.py` with the source log, source
header/blob, compiler log/build receipt, compiler bundle, and input identity
receipt to compare cycles and all 33 late top tokens.
