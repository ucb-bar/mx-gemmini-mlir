# Private BF16 VPU-to-matrix preparation transport

The explicit `--vpu-matrix-transport` route connects qualified BF16 producer
outputs to BF16-input activation preparation through Merlin's private-edge
rewrite. Complete compiled projection stages preserve every output word:

| Captured rows, matrix format | F32 transport cycles | BF16 transport cycles | Reduction |
|---|---:|---:|---:|
| 1, FP8 | 45,699 | 38,293 | 16.2060% |
| 19, FP8 | 151,378 | 135,919 | 10.2122% |
| 19, FP4 | 150,930 | 135,495 | 10.2266% |
| 1, FP8 with implicit zero rows | 14,649 | 14,649 | 0%: original route retained |

Each stage contains two broadcast multiplies feeding three independent linear
projections, with width 37 and output widths 23, 41 and 17. Row and column tails
are exercised through actual native matrix execution and ranked ABI shims.
The complete stage timer includes host padding, allocation, packing and dispatch.
These are small compiled stages, not complete model or cached-session timings.

Merlin owns view cleanup, edge privacy, liveness and representation rewriting.
MX registers exact compiled endpoint alternatives, layouts, buffers and completion
contracts. Only private synchronous calls with exact original catalog types can
change representation. Public outputs, actual fanout, unknown effects and
unsupported endpoints retain the original execution. The catalog/source/routing
bindings are reopened before selection. No model name, provenance label or
golden value selects the route.

The BF16 producer retains the intermediate rounding, finite range guard and
original fallback. BF16 preparation copies and zero-pads raw words, requiring
half the original padding scratch. The per-entry padding budget cannot grow.
The flag requires connected multiplies, shared activation packing and integer
word-group preparation. It defaults off. The current endpoint does not implement
implicit-zero BF16 rows; those preparations retain their original F32 route.
When no compact preparation is available, the builder emits neither unused
compact producers nor the BF16 quantizer entry. The last table row confirms
the retained route, identical outputs and unchanged measured cycles.

Four separate exceptional-boundary pairs cover NaN activation input, NaN weight
input, infinity input and finite overflow. All complete normalized output words,
including partial output before refusal, agree between actual F32 and BF16
producer executions. NaN inputs retain exit 77 from the test observer, RISC-V
cause 3, and an explicit `ebreak` at the actual reported instruction address.
Successful Inf/overflow cases retain their BF16 words. These pairs establish
boundary equivalence/refusal; they are not timing or independent quality results.
Packing's exhaustive finite/nonfinite and codebook checks are retained in
[bf16_activation_pack_20261019](../bf16_activation_pack_20261019/README.md).

Whole-process command/fence/load/store/scale counters match within each compiled
stage pair. Host tracking is off. Linked ELFs grow 53,504→55,728 bytes for one
row and 57,712→60,000 for 19 rows because alternative endpoints remain linked.
Literal allocation-site sums decrease by two bytes per private producer element;
these are not peak-memory measurements. The implicit-zero retained route has
the original 53,504-byte image size. Static padding buffers and both producer
alternatives remain separate from the stage's allocation-site sum.

```sh
python docs/evidence/vpu_matrix_transport_20261019/verify.py
```

The standalone verifier requires Python and NumPy, imports no compiler code,
and uses the pinned adjacent ELF reader. It reopens all sixteen actual images,
checks complete compiled stage outputs/digests/cycles and matched counters,
reopens exceptional-test payloads embedded in their ELFs, checks every observed
boundary word, guards, low-bit preservation and breakpoint instruction, and
binds original receipts/admissions/recipes. It checks 58,104 complete stage and
boundary words. The archive includes exact compiled sources/objects/images,
full logs/counters, transformed IR and rewrite receipts, captures including
checkpoint tensors, and relevant source at this checkpoint. It does not claim
complete Python/toolchain/startup-header dependency closure, a hermetic rebuild,
independent hardware instruction proof or a whole-model transport speedup.

Two first attempts are retained in the workspace. One encountered a catalog
namespace mismatch, fixed before the successful graph runs. Initial small
executions omitted explicit timing environment; the comparator refused their
missing timing counters. Their timings are not used. All table entries are fresh
executions with identical explicit `both/e4m3_vpu/mem.host_tracking=0` settings.

Reproduce a capture and matched builds in new task-local directories:

```sh
export GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu GEMMINI_PERF_SET=mem.host_tracking=0
python -m tools.capture_vpu_projection_probe --rows 19 --out-dir "$TMPDIR/reverse-probe" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
```

Build its `capture/` using `tools.build_merlin_stage`, the generated policy,
prepacked/pruned checkpoint weights, connected broadcast multiplies, prepared
broadcast inputs, shared activation packing, integer groups, structural ELF
admission and a complete output cap of 1,539. Add `--vpu-matrix-transport` only
to the candidate, then compare both with `tools.compare_shared_preparation`.
Use one row and an 81-word cap for the other shape. FP4 uses a separately saved
policy selecting FP4 for all three matrix sites in both variants. Add
`--virtual-zero-padding` to both one-row builds for the retained-route case.
Original recipes, source inputs and policy files are archived. The separate
`tools.qualify_compact_producer_boundary_spike` takes a checked broadcast capture,
explicit admission and `--expect-refusal` for NaN cases.

A matched complete 160-token control/candidate pair is running against the same
current packing, guarded-math and tile-selection sources. Complete logits and
the unchanged source-relative gate are required before reporting a new complete
model result for this transport.
