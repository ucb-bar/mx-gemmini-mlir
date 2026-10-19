# Full TinyLlama logit diagnostic

This archive contains one captured 128-token prefill and 32 decode
steps on the reference implementation’s `e4m3_vpu` Spike model. The compiler runs with 155 FP4
and 45 FP8 MX bindings per stage. The output-tap link uses the same compiled
stage objects and checkpoint weight blob as the completed 33-step compiler
timing run. It dumps output 0 after each timed step, so all 33 logits can be
compared against the captured unquantized FP32 references.

The guest completed all 33 steps and all **99 output digests match** the
earlier compiler timing run. Both runs record 132 expected and observed
session events. The tapped program uses the same stage objects and weight
blob as that run. Its raw modeled count is 83,547,479,772 cycles; the
previous untapped compiler count is 83,547,564,353 cycles. These counts
are not an independent reference-source model comparison.

| Compared with captured unquantized FP32 | Logit cosine | Normalized RMSE | Top-token agreement |
| --- | ---: | ---: | ---: |
| Prefill | −0.313245 | 1.255276 | token 5817 vs FP32 token 279 |
| 32 decode steps, mean | −0.323791 | 1.297424 | **0/32** |

The minimum decode-step cosine is −0.657827. The stream uses fixed synthetic
token IDs, so these are numerical diagnostics, not a generation-quality
measurement. The selected mixed FP4/FP8 policy does not preserve the FP32
logits on this input. The separate [precision control](../tinyllama_prefill_precision_20261014/README.md)
measured 0.943 cosine for an all-FP8 *prefill*; full-session accuracy under
that policy remains unmeasured.

The [model guide](../../compile_e2e_models.md#inspect-full-tinyllama-logits)
contains the relink, Spike, and comparator commands. `archive.py` stores the
completed outputs and log with hashes; `python -B verify.py` recomputes the
metrics from the archived bytes.
