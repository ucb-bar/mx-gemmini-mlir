# Independent TinyLlama source with VPU residual additions

This control adapts Nicolas's pinned `llama_model.c` by copying the VPU
residual-add command pattern from his `llama_layer_e2e.c`. The rest of the
source model keeps its own MX mesh and scalar-host schedule. The
[`vpu_residual.patch`](vpu_residual.patch) also adds an isolated full-stage
timer that excludes generated validation loops and logit grading. It changes
the source program; it does not substitute one MX issuer into a Merlin ELF.

The first measured instance has **two pretrained decoder layers plus the
final norm and head**. A separate [weight verifier](verify_weights.py)
confirms that all **20 FP32 checkpoint tensors** in the source and model2MLIR
captures are elementwise identical, as are all **64 token IDs**. Its
diagnostic run passed every generated mesh, host, and residual
seam check. The separately timed run produced **bit-identical final logits**
and took **366,920,372 functional Spike cycles** from residual-stream setup through
complete head readback. The compiled two-layer residual-add control took
**455,307,913 functional Spike cycles**. Their 32,000 final logits have **0.9958 cosine**,
both predict token **5056**, and the source has **0.9968 cosine** to the
captured PyTorch row. These figures are scoped to two layers and one input;
the receipt intentionally does not declare a full-model speedup ratio.
These two runs used the extension's default `func` mode, which does not
activate Nicolas's `e4m3_vpu` timing model. The reproduction script uses
`GEMMINI_MODE=both` for the full-depth performance run.
The source generator's own `grade logits vs torch` line refers to its
full-depth PyTorch capture; the cosine above uses the separate two-layer
PyTorch row from model2MLIR.

The [two-layer receipt](l2_receipt.json) pins the source and RTL commits,
input, Spike extension, program and blob hashes, output hashes, and both cycle
counts. The [diagnostic](diagnostic_l2head.log.gz) and
[timed](timed_l2head.log.gz) logs preserve the complete output. Run
`python verify_l2.py` with NumPy to check the archived logs against the
compiler control and PyTorch reference. To recheck capture identity, run
`verify_weights.py --source-capture <source-capture2> --compiler-stage
<model2mlir-two-layer-prefill>` with NumPy and safetensors. The
[diagnostic](diagnostic_l2_wrapper.c)
and [timed](timed_l2_wrapper.c) wrapper files show the build-mode macros.

Run the adjacent [full-source reproduction](../nicolas_tinyllama_full_source_20261014/reproduce.sh)
first, then use [`reproduce.sh`](reproduce.sh) in the same task-local
`WORK_DIR` to compile and run the VPU residual variant for all 22 layers.
That full-depth result is a separate measurement; the two-layer result here
does not establish its cycle count or model accuracy.
