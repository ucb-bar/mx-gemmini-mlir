# Full TinyLlama logit diagnostic

This directory is prepared for one captured 128-token prefill and 32 decode
steps on Nicolas's `e4m3_vpu` Spike model. The compiler runs with 155 FP4
and 45 FP8 MX bindings per stage. The output-tap link uses the same compiled
stage objects and checkpoint weight blob as the completed 33-step compiler
timing run. It dumps output 0 after each timed step, so all 33 logits can be
compared against the captured unquantized FP32 references.

The receipt will be added only after the Spike guest finishes and every
output digest matches the completed compiler run. A successful comparison
reports logit error for this fixed synthetic token stream. It does not
establish generation quality or exact MX quantized arithmetic correctness.
The stage timing is separate from console output and result hashing.

The [model guide](../../compile_e2e_models.md#inspect-full-tinyllama-logits)
contains the relink, Spike, and comparator commands. `archive.py` stores the
completed outputs and log with hashes; `verify.py` recomputes the metrics
from the archived bytes.
