# TinyLlama full-depth MX plus selected VPU control

This is a single 64-token prefill of the pretrained TinyLlama checkpoint: all
22 decoder layers, final norm, and language-model head. The model2MLIR capture
and MX compiler selected 200 MX contractions. The target bound every selected
contraction to FP8, prepacked checkpoint weights, and offloaded residual add,
unary, and reduction regions to the VPU. Softmax and other elementwise work
remained on the compiled host. Spike finished the stage in **4,473,524,179
functional-mode cycles**. The reference implementation’s `e4m3_vpu` timing model was not active.
This run does not include autoregressive decoding.

The final 32,000 logits have cosine **0.9594** to the captured PyTorch FP32
row and **0.9767** to the [independently generated FP8 source
row](../nicolas_tinyllama_full_source_20261014/README.md). All three rows
predict token **1230** for this one diagnostic input. This improves upon the
earlier compiler control that offloaded attention softmax and elementwise
adds; it does not establish language-model accuracy across inputs.
The [5% numerical gate](quality_gate.json) also fails for this one row:
the reference implementation’s source has KL **0.07090** nats to the FP32 logits, while the
compiler has KL **0.15506**. The allowed compiler KL is **0.07444**.
Matching the top token does not close this error gap. This control predates
the per-site execution policy; it is a full-depth diagnostic, not a
qualification of that newer policy.

The independent source program reported **4,456,723,475 cycles** in selected
MX and scalar-host windows. The two counters are **not a speedup comparison**:
the compiler counter covers the entire stage, while the source counter sums
selected windows; the compiler uses VPU residual/unary/reduce operations,
while the source uses scalar host operations. There is still no stitched,
independent MX+VPU full-model source baseline with matching operations and
timing boundaries.

The [receipt](receipt.json) binds the captured model, checkpoint weights,
precision map, profile, Spike extension, ELF, log, and complete logits to
their hashes. The compressed [Spike log](spike.log.gz) retains the output and
cycle count. Run `python verify.py` with NumPy to recheck the result against
the archived source and PyTorch references. Reproduction uses the captured
64-token TinyLlama prefill stage and `tools.build_merlin_stage` with
`--prepack-checkpoint-weights`, `--diagnostic-vpu-classes add,unary,reduce`,
the all-FP8 override map in the linked source evidence, and the pinned
MX+VPU Spike extension. The large ELF is regenerated locally.
