# Whole-model optimization checkpoint, 19 October 2026

The optimization goal remains active. The latest completed independent TinyLlama
forward uses **11,094,646,734 modeled cycles**, versus Nicolás's equal-boundary
12,762,360,525 and the corrected compiler's 15,984,649,026: reductions of 13.0674%
and 30.5919%, respectively. The older 15,170,996,572-cycle compiler archive had
different numerical quality and is not the equal-quality compiler control.

Every completed optimized forward preserves all 5,120,000 F32 logit words from
the corrected compiler. The unchanged source-relative gate passes on the 33
archived full source distributions: mean KL 0.0637956270, worst-row KL
0.1699797580, and 27/33 top-token agreement. All 160 compiler rows are checked;
full independent source distributions exist for only 33. Gate acceptance must
not be described as bit identity with the independent implementation.

| Independent 160-token forward | Cycles |
|---|---:|
| Corrected compiler placement | 15,984,649,026 |
| Shared activation packing | 14,452,067,124 |
| Connected sigmoid and matrix/BF16 handoff | 13,735,943,708 |
| Broadcast-connected RMS | 13,479,416,598 |
| Prepared broadcast operands | 13,215,457,143 |
| Generic word-group preparation | 11,197,470,681 |
| Exact guarded host math | 11,105,781,721 |
| Movement-bounded native tile selection | 11,094,646,734 |

The last tile change reduces measured load bytes from 4,198,111,552 to
3,939,506,496 and scale bytes from 115,279,872 to 107,198,464. Commands 3,312,204,
loops 38,608, fences 492,001, VPU commands 29,364 and mesh tiles 40,970,240 match.
These counters cover the wider process with host tracking off. Both images are
1,330,818,680 bytes; literal allocation sites sum to 1,372,484,654 bytes in 1,061
sites with no dynamic-sized sites. That sum is not peak memory. The selected
arena is 2 GiB in both builds.

Measurements bind specific archived sources, objects and images. A later generic
BF16-input helper refactor changes object bytes. Its fresh matched full-model
control/candidate pair is running; the table does not claim exact binary identity
with current HEAD. The historical OOT word-packing prototype is separately
archived and is not the generic Merlin implementation.

The eight requested areas currently have the following coverage:

| Area | Implemented or established | Remaining limits/work |
|---|---|---|
| Numerical placement | Intermediate diagnosis, corrected sensitive host placement, unchanged source-relative acceptance | Captured policies remain explicit; no automatic model-wide numerical-policy inference is claimed |
| Connected pointwise | Multiply pairs, sigmoid/multiply chains, projected broadcasts, prepared broadcast operands, exact intermediate rounding and fallback | General arbitrary pointwise DAG fusion is not implemented |
| Matrix/VPU/requant handoffs | Private matrix-to-VPU BF16 edges; reverse connected-VPU-to-shared-packing BF16 endpoints with graph/privacy checks | Reverse full-model pair is running; broader batched attention and accumulator-resident epilogues remain unqualified |
| Network scheduling | Shared preparations across independent consumers under checked effects, liveness and memory budgets | No asynchronous overlap or double buffering; native matrix code already batches loop commands without a CPU fence per loop |
| Weight/activation reuse | Source-proven immutable prepacking, unused entry-weight pruning, shared live activation preparation, BF16 read-view reuse | No cross-stage deduplication of encoded matrix constants or persistent encoded KV preparation |
| Prefill/decode/state | Separate captures and schedules, qualified virtual zero rows, complete cached-session comparison | State copies identified below; incremental KV packing requires ownership/update/version proof or exact content revalidation |
| Host/memory | Generic word-group conversion/max preparation, fixed-axis inlining, exact guarded library calls, bounded private buffers and session weight pools | Allocation sums are not peak measurements; no general temporal allocator or zero-copy state rotation is claimed |
| Cost selection | Actual matched native measurements, bounded tile search and resource guards, retained original routes | Experimental flags remain explicit; no general XLA-style fusion cost model is implemented |

Representative rejected or retained cases matter. A cold attention tile candidate
regressed despite a warm win, so its original tile remains. The measured decode
tile alternative also regressed and remains unselected. Out-of-domain host-math
fallback can regress by about 10.2%; the guard and original fallback are retained.
BF16 reverse transport retains F32 preparation for implicit-zero rows without a
qualified compact endpoint. Private-edge rewrites refuse public outputs, real
fanout, changed types, unknown effects or missing completion/layout contracts.

The completed 128-token prefill plus 32 decodes measures 74,065,304,284 cycles for
the corrected cached compiler control and 55,871,481,378 for the earlier reuse
passes plus virtual decode padding, a 24.5646% reduction. All 99 output/state
hashes and all 33 complete logit tensors match. This is a separate workload from
the independent forward and uses earlier passes, without the later RMS,
word-group, guarded-math, tile or reverse-transport changes. Its FP32 diagnostic
is identical across variants, but no independent Nicolás cached baseline or new
generation-quality threshold has been established.

Inspection of the actual generated cached adapter confirms that each decode
calls `merlin_commit_state` before returning. That runtime copies the entire
7,208,960-byte KV state and 176-byte length state into separate static input
arrays. The prefill-to-decode handoff also copies these tensors once. These
copies are inside the measured step/preparation boundaries. Their isolated cost
has not been measured. A future pointer-rotation change needs a generic lifetime
contract preserving output observation, reset, next-step input/output separation
and alias semantics. Pointer/shape/length alone cannot validate an encoded KV
cache for arbitrary input calls.

The completed SmolVLA session currently measures 89,671,723,299 → 87,964,565,915
cycles for equivalent BF16 read-view packing reuse (1.9038%). All 33 output/state
hashes and 11 full action tensors match. Its diagnostic final cosine is 0.9752;
there is no independently established full Nicolás SmolVLA baseline or action
acceptance threshold. Fresh controls and candidates for the newer passes have
compiled and are linking/executing. The new prefix selects 25 connected multiply
pairs, and the flow selects none. Both have zero eligible matrix/VPU transport
edges; selected flags alone must not be counted as applied transformations.

The session linker now carries an explicit exact guarded-math selection into
the final image and verifies its original library suppliers. A complete small
recurrent qualifier preserves every state/output hash and dumped word while
using 679,140 → 640,228 inclusive cycles. This is section/session integration
evidence, not the pending full SmolVLA result.

Primary retained evidence and standalone verification:

- [Generic packing full forward](evidence/matched_merlin_word_packing_20261019/README.md)
- [Guarded-math full forward](evidence/matched_guarded_math_forward_20261019/README.md)
- [Tile-selection full forward](evidence/matched_native_tiles_forward_20261019/README.md)
- [Complete cached sessions](evidence/cached_session_reuse_20261019/README.md)
- [Complete SmolVLA reuse session](evidence/smolvla_widening_session_20261019/README.md)
- [Reverse BF16 transport qualifiers](evidence/vpu_matrix_transport_20261019/README.md)
- [Guarded session integration](evidence/guarded_session_link_20261019/README.md)
- [Rejected/accepted native tile measurements](evidence/native_tile_search_20261019/README.md)

The initial changes came from profiling and inspecting generated IR. Subsequent
reference study covered the OpenXLA GPU architecture, IREE Stream/data tiling,
and IREE's DispatchCreation pipeline. The relevant design connections are
producer-consumer fusion, representation propagation, preparation hoisting and
explicit dependency/lifetime handling. No pass was ported, and XLA's
priority-fusion implementation was not reviewed. Generic transformations live
in Merlin; target encodings, layouts, capabilities and endpoint ABI live in MX.
