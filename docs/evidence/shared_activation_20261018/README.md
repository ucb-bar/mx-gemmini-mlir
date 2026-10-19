# Explicit activation packing reuse

This archive qualifies a compiler transformation on three-projection sections.
It does not establish full-model quality, cached-session speed, or an equal-quality
win against the independent source implementation.

| Section | Ordinary packs | Shared pack | Packing shared, unused RHS retained |
| --- | ---: | ---: | ---: |
| TinyLlama first-layer Q/K/V, 160 rows | 76,996,487 | 30,248,020 | 30,251,404 |
| Same projections, one source activation row | 7,012,079 | 2,819,959 | 2,820,652 |
| Independent 19×37 input, output widths 23/41/17 | 458,642 | 177,363 | — |

Every complete output word matches its compiled baseline: 409,600, 2,560 and
1,539 words respectively. The observed reductions are 60.715%, 59.784% and
61.329%. The ablation attributes essentially all cycle savings to shared packing:
retaining the unused RHS descriptors adds only 3,384 and 693 modeled cycles.
Removing those descriptors separately avoids large unused host allocations.
Device matrix commands and tile counts are unchanged.

| Resource | 160 rows | One row |
| --- | ---: | ---: |
| Explicit shared packed tensor | 337,920 B | 33,792 B |
| Static allocation estimate, ordinary → shared | 20,513,088 → 1,976,656 B | 18,884,928 → 44,368 B |
| Device commands, unchanged | 1,295 | 655 |
| Device load bytes, unchanged | 18,350,080 B | 5,439,488 B |
| Scale bytes, unchanged | 573,440 B | 169,984 B |

The allocation figures are compiler site estimates, not measured peak live
memory or reductions to the reserved runtime arena. Host packing traffic and
device load counters have different scopes.

The one-row section uses a real source activation and checkpoint, with physical
FP8 padding to 16 rows. It is not an autoregressive decode trajectory. The source
activations come from the FP32 first input normalization, not an intermediate
dump from the full quantized model. Source receipts verify checkpoint tensors
and original capture identities. Missing full-model source outputs are not inferred.

The independent tail fixture also ran with a 1,024-byte preparation budget,
smaller than its 2,112-byte prepared tensor. The pass retained ordinary execution:
458,642 cycles and all words unchanged. Generic tests cover effects, distinct SSA
inputs, static views, private destinations, overlapping lifetimes and byte budgets.
Native adapter tests cover both formats, finite extremes, changed inputs, padding,
repeated consumers and unchanged nonfinite refusal. The flag remains experimental;
there is no broadly qualified profitability policy yet.

Select `--share-activation-packing` with immutable prepacking and an explicit
execution policy. Merlin owns static view equivalence, effect barriers, consumer
rewriting and buffer budgets. MX owns the physical packing and consumer ABI.
`--retain-shared-activation-rhs` selects the attribution ablation.

The measured interval is the existing compiled stage forward interval, including
input preparation, padding, quantization, dispatch, matrix work, output widening,
host concatenation and temporary allocation. Initial harness setup and output
printing remain outside it. All runs use the same e4m3_vpu profile, extension,
`GEMMINI_MODE=both` and `GEMMINI_PERF_SET=mem.host_tracking=0`. These are modeled
cycles, not hardware measurements. The traffic counters in each comparison are
extension-wide counters and retain their wider scope.

`executions.tar.xz` retains actual admitted ELFs, complete consoles, comparisons,
captures excluding the large real-model FP32 checkpoint files, generated adapter
and issuer sources, typed rewrites and compiler receipts. Their identities remain
in the capture and source receipts. Compression shares repeated image content.
`verify.py` uses only the Python standard library and never extracts files. It
reopens every output word, image digest, accepted structural admission record and
cycle count. Structural admission records impose no instruction policy.

Run `python verify.py` to verify this archive. `reproduce.sh` describes fresh capture,
build and comparison commands using the explicitly selected existing toolchains.
