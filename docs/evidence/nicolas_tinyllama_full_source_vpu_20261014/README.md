# Independent TinyLlama source with VPU residual additions

This control adapts the reference implementation’s pinned `llama_model.c` by copying the VPU
residual-add command pattern from his `llama_layer_e2e.c`. The rest of the
source model keeps its own MX mesh and scalar-host schedule. The
[`vpu_residual.patch`](vpu_residual.patch) also adds an isolated full-stage
timer that excludes generated validation loops and logit grading. It changes
the source program; it does not substitute one MX issuer into a Merlin ELF.

## Full 22-layer prefill on modeled Spike

The [full-depth receipt](full_receipt.json) records one **64-token prefill**
through all 22 pretrained TinyLlama decoder layers, final norm, and head.
The reference implementation’s source and the model2MLIR/Merlin compiler capture use the same 64
token IDs and all **200 elementwise-identical FP32 checkpoint tensors**. The
regenerated source blob is byte-identical to the earlier independent source
control. The timed source computes each layer's activations from its preceding
layer; stored golden tensors are used only by disabled diagnostic checks. Its
VPU residual-add variant produces final logits **bit-identical** to the earlier
scalar-host source run.

| One 64-token prefill | Source MX+VPU | Compiler MX+VPU |
| --- | ---: | ---: |
| `e4m3_vpu` modeled stage cycles | 4,328,997,732 | 5,072,967,821 |
| Final-logit cosine to captured PyTorch | 0.9835 | 0.9546 |
| Final token | 1230 | 1230 |

Source-versus-compiler final-logit cosine is **0.9816**. The compiler uses
FP8 for all 200 MX contractions and routes 88 add regions to the VPU. The
source uses the reference implementation’s FP8 MX schedule and VPU residual-add pattern in 44
calls, while its normalization, RoPE, softmax, SiLU, and other host stages
remain handwritten. The compiler and source timers cover the same model
prefill but include different setup and host/VPU operations. The compiler's
raw counter is **743,970,089 cycles higher**; this is a diagnostic gap,
not a calibrated model speedup ratio. It is also one prefill, not the
33-step autoregressive TinyLlama session.
The [last-row numerical gate](quality_gate.json) fails: source KL to the
captured FP32 logits is **0.07090** nats and compiler KL is **0.19283**,
against a 5% limit of **0.07444**. Both predict token 1230. These numbers
measure this older add-only compiler control; the newer capture-bound policy
needs its own full-depth run.

Both binaries ran with `GEMMINI_MODE=both`,
`GEMMINI_PERF_CONFIG=e4m3_vpu`, `mem.host_tracking=0`, and
`--isa=rv64gc_zicntr` on the same pinned MX+VPU Spike extension. The Merlin
stage harness was updated to read the modeled `cycle` CSR; its two RV64
counter instructions are preserved in the
[disassembly](compiler_counter_disassembly.txt). The archived
[source](source_full_vpu.log.gz) and
[compiler](compiler_add_zicntr.log.gz) logs contain the complete output;
[source](source_perf_report.txt) and
[compiler](compiler_perf_report.txt) timing reports record MX/VPU activity.
Run `python verify_full.py` with NumPy to recheck the hashes, counters,
outputs, and timing settings. [`archive_full.py`](archive_full.py) accepts
the completed local Spike files and reproduces the small archived evidence.

## Two-layer numerical control

The first measured instance has **two pretrained decoder layers plus the
final norm and head**. A separate [weight verifier](verify_weights.py)
confirms that all **20 FP32 checkpoint tensors** in the source and model2MLIR
captures are elementwise identical, as are all **64 token IDs**. Its
diagnostic run passed every generated mesh, host, and residual
seam check. The separately timed run produced **bit-identical final logits**
and took **366,920,372 functional Spike cycles** from residual-stream setup
through complete head readback. The compiled two-layer residual-add control
took **455,307,913 functional Spike cycles**. Their 32,000 final logits have
**0.9958 cosine**,
both predict token **5056**, and the source has **0.9968 cosine** to the
captured PyTorch row. These figures are scoped to two layers and one input;
the receipt intentionally does not declare a full-model speedup ratio.
These two runs used the extension's default `func` mode, which does not
activate the reference implementation’s `e4m3_vpu` timing model. The reproduction script uses
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
The two-layer result is a separate numerical smoke test of the same VPU
residual-add pattern.
