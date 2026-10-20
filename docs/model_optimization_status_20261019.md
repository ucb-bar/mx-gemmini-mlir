# Completed model transformations, 19 October 2026

The newer [Radiance-derived Rocket comparison](evidence/radiance_rocket_optimization_20261019/README.md)
uses a 7,082,331,012-cycle full-forward control. The fastest completed compiler
placement takes 5,849,962,853 modeled cycles, but fails the unchanged full/late
top-token gates. No qualified Radiance speedup is established by that result.
The linked evidence preserves every completed alternative and its failures.

The fixed release below takes **10,326,818,367 modeled cycles**, versus
12,762,360,525 for the independent `npu-exploration` program: 1.236× speedup
with the same token-to-logits boundary and a passing source-relative late-33
gate. The corrected compiler baseline takes 15,984,649,026 cycles. The
fixed-release cumulative transformations below preserve its 5,120,000 F32
output words exactly.

| Cumulative forward build | Modeled cycles |
| --- | ---: |
| Corrected numerical placement | 15,984,649,026 |
| Shared activation packing | 14,452,067,124 |
| Connected sigmoid and matrix/BF16 handoff | 13,735,943,708 |
| Broadcast-connected RMS | 13,479,416,598 |
| Prepared broadcast operands | 13,215,457,143 |
| Integer word-group preparation | 11,197,470,681 |
| Exact guarded host math | 11,105,781,721 |
| Movement-aware native tile selection | 11,094,646,734 |
| Reverse VPU/BF16 preparation transport | 11,008,973,639 |
| Repeated batch RHS preparation reuse | 10,433,218,556 |
| Private compact RHS inputs | 10,326,818,367 |

These are cumulative measured builds, not independent percentage savings.
The old 15,170,996,572-cycle build failed numerical quality and is not an
accepted baseline. The compact forward image predates the later privacy cost
guard; separate reanalysis confirms all 44 selected uses are private. That
reanalysis does not add the guard to the historical executable.

Cached TinyLlama's complete 128-token prefill and 32 decodes now measure
**20,262,623,973 inclusive cycles**, versus the corrected compiler baseline's
74,065,304,284. The final guarded compact-RHS change reduces
23,669,399,808 to 20,262,623,973 cycles while preserving all 99 output/state
digests and 33 complete logit tensors. This actual cached image includes the
privacy guard. Its device work counters are unchanged; lower host preparation
cost accounts for the main improvement. Constant pooling has no qualified
full-session result included in this release.

SmolVLA's complete prefix, ten flow steps, and action decode measure
**73,577,503,143 inclusive cycles**, versus 87,964,575,985 for the previous
compiler build. Prepared broadcast operands and guarded host math preserve
all 33 output/state digests and 11 complete action tensors. Final action
cosine is approximately 0.9752 to the original floating-point trajectory;
no robot-task accuracy threshold is established.

The target-specific rewrites use graph and lifetime information for immutable
checkpoint prepacking, shared activation preparation, private BF16 boundaries,
and compact repeated operands. They are explicit supported transformations;
this release does not claim automatic inference of a model's numerical policy,
attention fusion throughout the network, or persistent encoded KV caching.
Unqualified or regressing alternatives are not selected by the published recipe.

The [comparison guide](fair_full_model_comparisons.md) gives all baseline
qualifiers. The [compilation guide](compile_e2e_models.md) provides exact
policies and reproduction commands. The [audit](model_results_audit.md)
reopens the retained artifacts rather than relying on this summary.

Selected complete-execution evidence:

- [Compact RHS forward](evidence/matched_compact_batch_rhs_forward_20261019/README.md)
- [Guarded compact RHS cached session](evidence/cached_compact_rhs_session_20261019/README.md)
- [Complete SmolVLA session](evidence/smolvla_latest_session_20261019/README.md)
- [Compaction cost guard and fallback](evidence/compact_batch_cost_guard_20261019/README.md)
