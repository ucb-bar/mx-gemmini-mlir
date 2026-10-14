# TinyLlama prefill precision diagnostic

This archive compares one captured 22-layer TinyLlama prefill and LM head
against its unquantized FP32 logits on fixed synthetic input IDs. All runs
use the same captured checkpoint weights and input. The completed mixed
FP4/FP8 full-session compiler run supplies the expected prefill output hash;
a separate output-tap run supplied its 32,000 values. The separate all-FP8
compiler prefill completed on Nicolas's `e4m3_vpu` Spike extension. The
five operand proxies use current model2MLIR revision `9153d34` and do not
model MX mesh accumulation or VPU arithmetic.

| Precision policy | Evaluator | Logit cosine vs FP32 | Top token | FP32 token rank |
| --- | --- | ---: | ---: | ---: |
| 155 dense FP4 + 45 attention/rotary FP8 | Spike | **−0.313** | 5817 | 21944 |
| Same mixed policy | Operand proxy | 0.498 | 800 | 40 |
| Promote final LM head only to FP8 | Operand proxy | 0.535 | 983 | 42 |
| Promote first 11 layers' 77 dense sites | Operand proxy | 0.805 | **279** | **1** |
| Promote last 11 layers and head's 78 dense sites | Operand proxy | 0.663 | 631 | 46 |
| All 200 contractions FP8 | Operand proxy | 0.922 | 362 | 2 |
| All 200 contractions FP8 | Spike | **0.943** | 362 | 3 |

The FP32 top token is 279. Head-only promotion does little; promoting all
ordinary dense sites greatly improves the logit cosine. Promoting only the
first 11 layers selects the FP32 top token in the operand proxy despite a
lower cosine than the all-FP8 control. The disjoint half-model recipes and
arrays are pinned by the [half-sweep receipt](half_sweep_receipt.json). They
need a Spike run before a compiler policy can be judged. The capture's MX
worklist maps their 77 and 78 source regions to layers 0–10 and layers
11–21 plus `lm.lm_head`, respectively. The negative mixed Spike cosine
also differs sharply from the mixed operand proxy. FP4 mesh
arithmetic, VPU rounding, or their interaction may account for that gap;
these measurements alone do not isolate the cause. Neither Spike policy
selects the FP32 top token. They establish no model-quality threshold.

The [receipt](receipt.json) records full precision, capture, and artifact
hashes. The mixed prefill bytes match the first output digest of the
[completed full compiler run](../nicolas_matrix_substitution_tinyllama_full_20261014/compiler/mx_session_run.json).
The all-FP8 Spike log and operand-proxy outputs are included so the metrics
can be recomputed without the original temporary build folders:

```sh
python3 -B docs/evidence/tinyllama_prefill_precision_20261014/verify.py
python3 -B docs/evidence/tinyllama_prefill_precision_20261014/verify_half.py
```

This is a **prefill diagnostic**, not a 33-step logit comparison or an
independent full-model stitched-kernel performance result. Those comparisons
require separate evidence bundles.
