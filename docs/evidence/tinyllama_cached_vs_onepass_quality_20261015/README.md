# TinyLlama cached-session numerical comparison

The archived compiler run executes **128-token prefill plus 32 cached decode
steps**. The reference implementation’s independent source run executes one causal forward over
the **same 160 token IDs** and emits its final 33 logit rows. The archived
[prefill](prefill_inputs.npz) and [decode](decode_inputs.npz) input files
concatenate exactly to the source's input IDs. The two FP32 references agree
at all 33 top tokens, have mean KL `2.7e-8` nats, and differ by at most
`2.67e-5` in a logit. This makes the rows a useful numerical cross-check.

| 33 aligned rows versus cached FP32 | reference source | Compiled cached session |
| --- | ---: | ---: |
| Mean KL, nats | 0.06417 | 1.92086 |
| Worst-row KL, nats | 0.18960 | 3.56428 |
| Correct top token | 27/33 | 2/33 |

The compiler **fails** the 5% numerical gate. The [receipt](receipt.json)
records the input and output hashes, reference agreement, and all three
checks. Run `python verify.py` to recompute them from the existing
[compiler session](../tinyllama_all_fp8_full_session_20261014/README.md)
and [reference source](../nicolas_tinyllama_onepass160_same_input_20261014/README.md)
archives; it needs NumPy and the MX Python package, but no checkpoint or
Spike run.

This comparison measures output quality on fixed synthetic tokens. The
source is a single 160-token forward; the compiler is a cached session, so
their cycle counts are **not** a paired performance comparison. The compiled
session used the older all-FP8, all-eligible-VPU route. The newer per-site
policy has only passed a two-layer Spike control and has not yet produced a
full cached-session result.
