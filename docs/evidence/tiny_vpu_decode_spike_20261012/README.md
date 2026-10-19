<a id="tinyllama-vpu-decode-on-nicolass-spike-2026-10-12"></a>

# TinyLlama VPU decode on the pinned Spike model (2026-10-12)

The full-checkpoint TinyLlama decode stage, compiled from the receipt-bound
model2MLIR capture, completed as a standalone RV64 program on the reference implementation’s
pinned MX+VPU Spike extension. It routed 200 source-bound MX contractions,
emitted guest `DONE`, reported zero memref rank mismatches, and produced
validated SHA-256 digests for its 32,000 logits, KV cache, and position state.
`index.json` records the capture, ELF, simulator, extension, run receipt,
cycle count, and three result digests. This one-step run uses the captured
decode inputs; it is not a recurrent step fed by a new compiled prefill.

The unquantized PyTorch golden is a diagnostic, not the numerical acceptance
reference for FP4/FP8 MX execution. Its first logit is -9.377924 and its
top index is 29871; the Spike output's first logit is -3.6875 and its top
index is 27305. The difference needs an independent model reference using
the compiler's exact FP4/FP8 operand conversion and BF16 VPU arithmetic.
Until that comparison exists, this result proves integrated execution and
output/state ABI coverage, not numerical correctness. A matched-capture run
of the earlier pre-VPU decode ELF is in progress to isolate the effect of the
new lowering.
