# TinyLlama all-FP8 prefill: safe and full VPU routes

Both standalone Spike runs use the same 128-token TinyLlama capture,
checkpoint, 200 direct E4M3 MX matrix bindings, target profile, and timing
settings. The safe route lowers 88 adds, 45 reductions, and 156 unary
regions to VPU; the full route additionally lowers 358 elementwise regions
and 22 softmax regions.

| VPU route | Modeled cycles | Top logit token | Output hash |
| --- | ---: | ---: | --- |
| Safe | 10,804,276,937 | 362 | `71116f11…` |
| Full | 12,276,460,508 | 29889 | `8ebd26b5…` |
| Captured FP32 | — | 279 | — |

The timer boundary and Spike configuration match across the two standalone
runs. The safe route takes 12.0% fewer modeled cycles, **but the full output
hashes differ**. This is a route-cost diagnostic, not an equal-output
speedup. The runs emit output hashes and top tokens rather than full logit
arrays, so this archive does not assign a cosine to the safe route. The
full-VPU output matches the linked all-FP8 prefill byte-for-byte; that
[full-session archive](../tinyllama_all_fp8_full_session_20261014/README.md)
reports prefill cosine 0.774153 to the captured FP32 logits.

The [comparison receipt](comparison.json) binds both stage runs to the
captured source and the same FP8 MX catalog. Run `python -B verify.py` to
check the compact evidence against the linked full-session archive without
the multi-gigabyte checkpoint or ELFs.
The copied receipts retain absolute paths from the measured build; the
verifier uses the archived receipts, logs, and baseline archive.
