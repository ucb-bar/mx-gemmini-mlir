# Independent TinyLlama source run on Spike

The pinned NPU source generator produced an FP8, 64-token TinyLlama program
with all 22 decoder layers and the final norm and language-model head. It used
the same checkpoint and token IDs as the model2MLIR 64-token capture. The
captured tokens are a diagnostic input, not a language-quality benchmark. The
run is one prefill forward pass; it does not include autoregressive decoding.
The source program passed its generated bit-exact checks for every mesh section,
host section, and boundary between them on the pinned MX Spike extension.

| Source `read_cycles` window | Cycles |
| --- | ---: |
| MX mesh sections | 69,364,062 |
| Scalar host sections | 4,387,359,413 |
| Sum of selected sections | 4,456,723,475 |

The compiler's independent 64-token, 22-layer, all-FP8 run reported
5,369,620,807 **stage** cycles. This is a raw counter, not a paired speedup:
the source times selected sections, uses scalar host normalization, RoPE,
softmax, SiLU, and residual operations, while the compiler times its full
stage and offloads selected operations to the VPU. In addition, their final
logits are not yet numerically equivalent. The cosine similarity of compiler
and source logits is 0.6720. The source logits have cosine 0.9835 to the
captured PyTorch reference; compiler logits have cosine 0.7011. Matching
argmax for this one synthetic input does not establish model accuracy.

A later [full-depth selected-VPU compiler
control](../tinyllama_full_safe_vpu_20261014/README.md) keeps the same MX
contractions, checkpoint, and FP8 policy but leaves softmax and elementwise
work on the compiled host. It has 0.9767 cosine to these source logits and
the same argmax, 1230. Its 4,473,524,179 stage cycles still cannot be divided
by the source's selected-window cycles to claim a speedup.

The [receipt](receipt.json) pins the source/RTL commits, input, generated
data, Spike extension, log hashes, 64-bit cycle counters, and comparison
scope. [Output comparison](output_comparison.json) records the complete-logit
metrics. The compressed [source](source_spike.log.gz) and
[compiler](compiler_spike.log.gz) Spike logs retain the raw output and cycle
lines; the [captured PyTorch logits](pytorch_last_logits.npy) provide the
independent FP32 reference. The receipt records all three hashes. The source
run uses `llama_model.c` from NPU exploration at the
receipt's source commit, with the [diagnostic print patch](diagnostic_prints.patch)
and `LLAMA_DUMP_LAST_LOGITS` enabled. The patch only prints 64-bit cycle
counters and the final BF16 logits; the generated model data is the
source generator's output for 22 layers plus the head. The data and ELF are
large and are regenerated locally rather than stored in Git.
The compiler's [all-FP8 override map](all_fp8_overrides.json) is archived and
bound to the captured model by the receipt.
The output comparison was first computed from a run whose source cycle line
printed 32-bit integers. The archived source log repeats the same outputs and
checks with the 64-bit diagnostic print patch; use its counters in the receipt
for timing.

Run `python verify.py` in this directory to recheck the archived complete
logits, output hashes, and raw cycle counters. It requires NumPy and does not
need the generated data blob or the Spike ELF.

The [reproduction script](reproduce.sh) applies the capture and diagnostic
patches, captures all 22 layers with the supplied model2MLIR input IDs,
generates the `_m64` model blob, builds with `LLAMA_DUMP_LAST_LOGITS`, and
runs Spike with `--isa=rv64gc_zicntr`. It requires a fresh `WORK_DIR` with
enough space for the generated data and ELF. The receipt hashes check the
input IDs, generated blob, and Spike extension.

This run does not substitute for a stitched full-model MX+VPU source baseline.
That baseline must use matching operations, shapes, precision, and timing
boundaries before a model-level cycle ratio is meaningful.
