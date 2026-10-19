# Whole-model optimization checkpoint, 19 October 2026

The optimization goal remains active. The latest completed independent TinyLlama
forward uses **10,326,818,367 modeled cycles**, versus Nicolás's equal-boundary
12,762,360,525 and the corrected compiler's 15,984,649,026: reductions of 19.0838%
and 35.3954%, respectively. The older 15,170,996,572-cycle compiler archive had
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
| Reverse connected-VPU/BF16 preparation transport | 11,008,973,639 |
| Source-proven repeated batch RHS preparation | 10,433,218,556 |
| Private compact RHS source inputs | 10,326,818,367 |

The last tile change reduces measured load bytes from 4,198,111,552 to
3,939,506,496 and scale bytes from 115,279,872 to 107,198,464. Commands 3,312,204,
loops 38,608, fences 492,001, VPU commands 29,364 and mesh tiles 40,970,240 match.
These counters cover the wider process with host tracking off. Both images are
1,330,818,680 bytes; literal allocation sites sum to 1,372,484,654 bytes in 1,061
sites with no dynamic-sized sites. That sum is not peak memory. The selected
arena is 2 GiB in both builds.

Measurements bind specific archived sources, objects and images. A later generic
BF16-input helper refactor changes object bytes. Its fresh matched full-model
pair completed: 11,094,643,214 → 11,008,973,639 cycles (0.77217%). The control
differs by 3,520 cycles from the historical native-tile row. The candidate
selects 44 private edges, with unchanged device counters and all logit words.
Literal allocations sum to 1,343,648,814 bytes, 28,835,840 fewer; this is not
peak memory. Its image is 1,330,837,896 bytes. The table binds frozen binaries,
without claiming exact binary identity with current HEAD. The historical OOT word-packing prototype is separately
archived and is not the generic Merlin implementation.

The eight requested areas currently have the following coverage:

| Area | Implemented or established | Remaining limits/work |
|---|---|---|
| Numerical placement | Intermediate diagnosis, corrected sensitive host placement, unchanged source-relative acceptance | Captured policies remain explicit; no automatic model-wide numerical-policy inference is claimed |
| Connected pointwise | Multiply pairs, sigmoid/multiply chains, projected broadcasts, prepared broadcast operands, exact intermediate rounding and fallback | General arbitrary pointwise DAG fusion is not implemented |
| Matrix/VPU/requant handoffs | Private matrix-to-VPU BF16 edges; reverse connected-VPU-to-shared-packing BF16 endpoints with graph/privacy checks and qualified full-forward improvement | Broader batched attention and accumulator-resident epilogues remain unqualified |
| Network scheduling | Shared preparations across independent consumers under checked effects, liveness and memory budgets | No asynchronous overlap or double buffering; native matrix code already batches loop commands without a CPU fence per loop |
| Weight/activation reuse | Source-proven immutable prepacking, unused entry-weight pruning, shared live activation preparation, BF16 read-view reuse, exact repeated RHS packing with qualified full forward and cached session; private compact inputs with qualified full forward; explicit cross-stage immutable pooling passes native/recurrent qualifiers and full-image admission | Compact RHS and pooled full cached executions remain pending; no persistent encoded KV preparation |
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

Two resident scalar-epilogue probes now retain the original route. A native
compute plus store-only scratchpad loop produces correct finite-case words,
but its timing counters account for only one of three expected matrix/VPU
executions, so its cycle comparison is refused. The existing explicit
scratchpad schedule has complete matching arithmetic counters and exact words,
but costs 4,550 → 5,693 section cycles (25.1209% more) despite 18,432 → 12,288
load bytes. Both use separate actual admitted images and three distinct inputs;
neither qualifies a general or full-model resident epilogue. At that checkpoint,
larger shapes, broader epilogues and a complete native handoff were unmeasured.

A subsequent explicit scale-state handoff now completes the native timing work
across repeated calls. Two unmanaged scale configurations after readout release
the managed halves; no simulator/profile change is made. Five finite-integer
section pairs preserve every output word and guard, with complete matching mesh,
VPU and scale work: 16×16×32 measures 2,052→1,796 cycles; 16×64×64
4,625→4,445; 32×32×64 4,550→4,403; 64×16×256 13,717→13,196; and
64×64×96 14,799→14,112. Each sums three changing-input invocations. Load bytes
fall by exactly three BF16 output-tensor reloads. This advances the resident
feasibility probe; there is still no production rewrite or whole-model resident
gain. Nonzero-K alternatives without the handoff retain incomplete work and an
explicit timing-model deadlock. New execution/comparison gates reject that
diagnostic even with exit-zero complete functional outputs; 58 focused tests
pass. All 91 completed stage logs and 22 completed session logs at this
checkpoint pass the diagnostic audit. See the
[native scale-handoff evidence](evidence/native_resident_scale_handoff_20261019/README.md).

An explicit cross-stage packed-constant pool now has generic Merlin ownership
and an MX native reader/relocation contract. The real latest prefill/decode
catalogs contain 310 byte-identical code/scale blob pairs, totaling 1,066,745,856
duplicate payload bytes. A full-size proof reconstructs both original catalog
objects exactly and recompiles all 625 selected reader/issuer objects per stage
to exact original bytes. Small native qualifiers cover partial tiles, changed
inputs, complete outputs/canaries, admitted images and matching work counters;
unknown observers and source/header changes refuse. The corrected full cached
link now passes admission and actual ELF/section checks: **2,439,193,888 →
1,373,933,592 bytes**, with the same 2 GiB arena and memory span. The simulation
is live; full output/quality/timing remains pending. A completed small recurrent
session preserves every output/state word and measures 640,228 → 640,134 cycles,
but its file grows 16,408 bytes despite 1,088 fewer constant bytes. This exposes
small-shape layout overhead; payload equality alone does not establish resource
profitability. See [pooling contract](shared_catalog_constants.md).

The completed 128-token prefill plus 32 decodes measures 74,065,304,284 cycles for
the corrected cached compiler control, 55,871,481,378 for the earlier reuse
passes plus virtual decode padding, and **41,567,160,094 for newer passes**.
The latest is 25.6022% below the earlier reuse session and 43.8777% below the
original corrected cached compiler. Inclusive prefill costs 8,885,018,358 cycles,
32 decodes 32,677,635,948, and reset 4,505,788. All 99 output/state hashes and
all 33 full logit tensors match exactly, including every actual dump and console
HEX stream. Its FP32 diagnostic is unchanged: mean KL 0.07095571052388557,
worst KL 0.18586350146524108, top agreement 28/33. No independent Nicolás cached
baseline or generation-quality threshold has been established. This separate
workload retains legacy native tiles; its latest completed result includes
broadcast/RMS, integer word packing, guarded math and private BF16 transport,
before the repeated/compact RHS changes described below.

The latest cached result reduces process-wide commands 9,797,152 → 9,329,686,
fences 182,241 → 181,785 and load bytes 37,556,271,872 → 37,493,680,896. Mesh
work and scale bytes match; VPU commands increase 42,730 → 42,754. The image
grows 5,288,040 bytes and static catalog BSS grows 5,416,512 bytes. Prefill/decode
literal allocation sums decrease 133,699,068/869,884 bytes, respectively; these
are not peak-memory measurements. Both keep the same 2 GiB arena and native
workspaces. The standalone reader and actual image/section/stage-object checks
passed; new evidence is 4,830,056 compressed bytes with a pinned dependency on
the earlier complete cached archive.

The ordinary repeated-RHS cached experiment has now completed: **41,567,160,094
→ 23,669,399,808 inclusive cycles (43.0575% fewer)**, or 68.0425% below the
original corrected cached compiler. The 32 decodes improve 54.7707%:
32,677,635,948 → 14,779,875,659 cycles. Identical prefill stage objects measure
8,885,018,358 → 8,885,018,361 (+3 cycles); reset stays 4,505,788. Every one of
the 99 state/output hashes and 1,056,000 full logit words remains exact, and the
FP32 diagnostic is unchanged. Commands, loops, fences, arithmetic and transfer
counters match; inclusive instructions fall 36,479,634,801 → 18,581,902,169.
The image grows 12,072 bytes, with unchanged allocations/arena/native workspace.
Both actual images, source-bound selection and standalone evidence verification
passed. The guarded compact-input session remains live; the newer constant-pool
implementation is a separate pending experiment, absent from this frozen image.

Inspection of the actual generated cached adapter confirms that each decode
calls `merlin_commit_state` before returning. That runtime copies the entire
7,208,960-byte KV state and 176-byte length state into separate static input
arrays. The prefill-to-decode handoff also copies these tensors once. These
copies are inside the measured step/preparation boundaries. Their isolated cost
is now measured using the actual compiled runtime on synthetic full-size bytes:
4,506,607 cycles per call, and 144,210,804 cycles for 32 calls (approximately
0.2581% of the completed 55.871-billion-cycle cached session). This is attribution,
not an exact subtraction from session time. A future pointer-rotation change needs a generic lifetime
contract preserving output observation, reset, next-step input/output separation
and alias semantics. Pointer/shape/length alone cannot validate an encoded KV
cache for arbitrary input calls.

Repeated batch RHS packing is the larger measured opportunity. A generic Merlin
coordinate proof recognizes equal adjacent matrices through pure singleton
broadcasts, linear views and matrix-axis transposes. The target retains private
packing within one synchronous call and refreshes every group on each invocation.
It does not remove source broadcasts or device transfers. QK/PV native sections
reduce cold cycles from 15,111,468 to 2,275,500 and 15,376,044 to 2,537,777,
respectively, with unchanged workspace and device counters. FP4/FP8 tails,
changed groups across calls and complete partial-output refusal cases qualify.
A complete random-F32 broadcast/transpose graph measures 1,118,895 → 765,994
cycles, all 2,622 output words exact, with 40 additional literal allocation bytes
and unchanged native workspace. This is not a whole-model speedup.

Both the real forward and cached decoder prove 44 eligible current calls across
two shared declarations. Their initial new-pass builds stopped before simulation
because the adapter incorrectly required one declaration per source call. The
corrected generic and target binders independently prove selected calls, validate
each shared native kernel family and retain original unproved users. The fresh
ordinary full-forward run completed: **11,008,973,639 → 10,433,218,556 cycles**
(5.22987% fewer), with all 5,120,000 logit words identical and the original source
quality gate unchanged. Device work/transfer counters and literal allocation
sums are unchanged. The image grows by 12,072 bytes, with the same 10,579,968-byte
native workspace and 2 GiB arena. Failed attempts remain preserved. The cached
candidate reuses the exact latest prefill stage and adds RHS reuse only to its
decoder; its complete result is recorded above. The matched cached control
already includes newer packing, broadcast, guarded-math and private BF16
transport passes.

A further optional compact-input ABI removes the need to construct the expanded
RHS for private consumers. Merlin supplies one source matrix per proved group,
including a compact transpose when necessary; target code reads matrix
`batch / repeat`. All original rounding, packing order, private workspace and
synchronous completion remain. Other consumers keep their original tensors;
profitability depends on the actual remaining uses. Explicit flags remain off
by default. A complete tail graph improves another 4.4939% (765,994 → 731,571
cycles), while a complete one-token QK graph improves 56.9763% (5,504,459 →
2,368,224), with all output words and device counters identical. Literal
allocations sum to 51,568 → 17,454 bytes and 2,642,152 → 184,478 bytes,
respectively; these are not peak measurements. Equal sample values without a
source proof select zero calls and preserve the original 1,091,675 cycles.
Seven native compact-ABI cases qualify tails, repeated calls and partial
refusals. Both real forward/decode graphs admit 44 compact calls, reducing RHS
batch32 to4. The complete compact forward now measures **10,433,218,556 →
10,326,818,367 cycles** (another 1.01982%), with all words and the unchanged source
quality gate preserved. It removes 55,379,622 literal allocation bytes across
23 sites (not peak memory), and shrinks the image by 61,056 bytes. The native
workspace, arena and device work/transfer counters remain unchanged. This frozen
image predates the later privacy guard; fresh current source reanalysis proves
all 44 of its selected uses private, but does not put a guard into that binary.
The guarded cached-session result remains pending.

A measured live-output counterexample now guards compaction: returning the
expanded RHS regresses 778,849 → 782,715 cycles if compacted unconditionally.
Generic fixed-point source-use closure rejects that case, and the target retains
its separately qualified ordinary packing-reuse endpoint at exactly 778,849
cycles. The private case retains 731,571 cycles. Complete first-output words and
independent full-byte/hash checks cover the returned RHS; missing or changed
additional references refuse. All 44 real forward/decode compact candidates
remain private under this guard. The fresh guarded compact decoder has compiled
and its complete cached session is simulating. Its actual final image passed
admission. Identical immutable linked checkpoint-object storage is shared only
after fresh production and complete byte verification; a small complete recurrent
qualifier preserves the exact image, outputs and 640,228 inclusive cycles. This
storage optimization enables the full run without changing execution semantics.

The newest complete SmolVLA matched pair measures **87,964,575,985 → 73,577,503,143
cycles (16.3555% fewer)**. Both already use the earlier BF16 read-view packing
reuse. The candidate adds word-group preparation/inlining, broadcast connections
and exact guarded host math. All 33 output/state hashes and 11 complete action
tensors match. The prefix selects 25 connected multiply pairs, and flow selects
none; both have zero connected sigmoid or matrix/VPU transport edges. The bundle
removes 307,200 commands, 4,800 fences and 39,321,600 load bytes with unchanged
mesh/VPU arithmetic work. The image grows by 8,366,768 bytes, while prefix literal
allocation sums fall by 78,645,300 bytes; these sums are not peak memory. Its
diagnostic final cosine remains 0.9752. There is no independently established
full Nicolás SmolVLA baseline or action acceptance threshold. The earlier
89,671,723,299 → 87,964,565,915 widening-only pair remains separate evidence.

Replaying the current repeated-RHS proof on the completed SmolVLA source selects
zero of 56 prefix and zero of 32 flow batch calls. Widening and non-matrix-axis
permutations interrupt its current exact-copy proof. Full source traces retain
repeated broadcasts behind some of those operations, so this is a present
eligibility limit, not a proof that broader reuse is impossible. Supporting
them needs a new coordinate/conversion proof and matched execution; no additional
SmolVLA batch-reuse result is claimed.

The session linker now carries an explicit exact guarded-math selection into
the final image and verifies its original library suppliers. A complete small
recurrent qualifier preserves every state/output hash and dumped word while
using 679,140 → 640,228 inclusive cycles. This is section/session integration
evidence; the full SmolVLA bundle result is reported separately above.

Primary retained evidence and standalone verification:

- [Generic packing full forward](evidence/matched_merlin_word_packing_20261019/README.md)
- [Guarded-math full forward](evidence/matched_guarded_math_forward_20261019/README.md)
- [Tile-selection full forward](evidence/matched_native_tiles_forward_20261019/README.md)
- [Reverse BF16 full forward](evidence/matched_reverse_transport_forward_20261019/README.md)
- [Repeated RHS preparation full forward](evidence/matched_batch_rhs_forward_20261019/README.md)
- [Private compact RHS full forward](evidence/matched_compact_batch_rhs_forward_20261019/README.md)
- [Complete cached sessions](evidence/cached_session_reuse_20261019/README.md)
- [Complete cached session with newer passes](evidence/cached_latest_session_20261019/README.md)
- [Repeated-RHS complete cached session](evidence/cached_rhs_session_20261019/README.md)
- [Resident matrix epilogue counterexamples](evidence/native_resident_epilogue_rejections_20261019/README.md)
- [Native resident scale-state handoff and timing rejection](evidence/native_resident_scale_handoff_20261019/README.md)
- [Immutable constant-pool native qualifiers](evidence/shared_catalog_constants_native_20261019/README.md)
- [Complete recurrent pooling and full-image admission](evidence/shared_catalog_constants_session_20261019/README.md)
- [Complete SmolVLA reuse session](evidence/smolvla_widening_session_20261019/README.md)
- [Complete SmolVLA preparation/host bundle](evidence/smolvla_latest_session_20261019/README.md)
- [SmolVLA repeated-RHS source applicability](evidence/smolvla_rhs_applicability_20261019/README.md)
- [Semantic placement and provenance-label regression](evidence/semantic_policy_placement_20261019/README.md)
- [Generic attention dataflow ownership and source compatibility](evidence/attention_dataflow_ownership_20261019/README.md)
- [Reverse BF16 transport qualifiers](evidence/vpu_matrix_transport_20261019/README.md)
- [Guarded session integration](evidence/guarded_session_link_20261019/README.md)
- [Rejected/accepted native tile measurements](evidence/native_tile_search_20261019/README.md)
- [Repeated batch RHS native and graph qualification](evidence/repeated_batch_rhs_20261019/README.md)
- [Compact batch RHS native, complete graph and fallback qualification](evidence/compact_batch_rhs_20261019/README.md)
- [Measured compaction regression and guarded fallback](evidence/compact_batch_cost_guard_20261019/README.md)
- [Verified linked-weight storage and full cached admission](evidence/linked_weight_storage_20261019/README.md)
- [Actual runtime state-copy attribution](evidence/state_copy_attribution_20261019/README.md)

The initial changes came from profiling and inspecting generated IR. Subsequent
reference study covered the OpenXLA GPU architecture, IREE Stream/data tiling,
and IREE's DispatchCreation pipeline. The relevant design connections are
producer-consumer fusion, representation propagation, preparation hoisting and
explicit dependency/lifetime handling. No pass was ported, and XLA's
priority-fusion implementation was not reviewed. Generic transformations live
in Merlin; target encodings, layouts, capabilities and endpoint ABI live in MX.

An architecture audit found and corrected a default-placement label heuristic:
elementwise adds were identified by `#add` in a provenance label. Placement now
uses the selector's verified arithmetic kind. Twelve renamed/misleading-label
cases reproduce three old failures and pass after the fix; the focused suite
passes 37 tests. Complete default policies remain identical on four archived
captures plus the current full TinyLlama and both SmolVLA stage captures. This
does not rebuild or change any measured or running image.

The target-independent attention relationship analysis has also moved from MX
into Merlin's frontend layer. The MX compatibility function delegates; physical
and numerical legality remain separate. The non-docstring analysis AST and all
ordered relationships on seven retained captures are unchanged. Thirteen generic
graph tests and 39 target integration tests pass. This is an ownership correction,
without a new fusion or runtime-performance claim.
