# Complete cached TinyLlama session with repeated-RHS preparation reuse

Source-proven repeated RHS preparation reduces the complete 128-token prefill
plus 32 cached decodes from **41,567,160,094 to 23,669,399,808 modeled cycles**:
**43.0575% fewer** than the latest compiler control and 68.0425% fewer than the
original corrected cached compiler's 74,065,304,284 cycles. Both processes
completed with exit code zero. All 99 output/state hashes and all 33 complete
logit tensors match exactly.

| Inclusive interval | Latest control | RHS reuse |
|---|---:|---:|
| Reset | 4,505,788 | 4,505,788 |
| Prefill plus preparation | 8,885,018,358 | 8,885,018,361 |
| 32 decodes plus state handoff | 32,677,635,948 | 14,779,875,659 |
| **Total** | **41,567,160,094** | **23,669,399,808** |

Decode costs **54.7707% less**. The prefill uses the exact same stage objects;
its interval changes by three cycles in the different linked image. Reset and
preparation intervals are unchanged. The original timing boundary includes
reset, program handoffs and measured step intervals, excluding console output
and phase bookkeeping. This is a complete cached session, separate from the
independent 160-token forward and kernel/issuer sections. There is no matched
independent Nicolás cached baseline.

Merlin proves exact contiguous RHS repetition from source coordinates. The
target packs one representative per repeated group into private call-local
preparation and reuses that representation for the remaining consumers. All
44 selected source calls per decoder invocation have batch 32 and repeat eight;
the 22-layer QK and PV families each contribute 22 calls. Two qualified native
helper kernels replace repeated preparation. Existing matrix schedules,
precision, rounding, source outputs and synchronous completion remain unchanged.
Other consumers retain the original tensors. This is ordinary RHS reuse, not
compact-input elimination or persistent caching across decode steps.

## Numerical evidence

The reader independently reopens every actual output file and its complete
console HEX encoding: 33 × 128,000 bytes, **1,056,000 F32 logit words**. Every word
matches the latest completed compiler control. All 99 ordered hashes also cover
KV state and position after each step.

The captured FP32 diagnostic is unchanged: mean KL 0.07095571052388557,
worst-row KL 0.18586350146524108, minimum cosine 0.9724467380582315 and 28/33
top-token agreement. These diagnostics do not establish a new generation-quality
threshold or independent source equivalence. No separate fast-run observation
is fabricated; the actual receipt status remains
`complete_numeric_qualification_pending`.

## Attribution and resources

Inclusive retired instructions fall **36,479,634,801 → 18,581,902,169**. The
17,897,760,286-cycle saving equals 17,897,732,632 fewer instructions plus the
27,654-cycle host-stall decrease. Thus the large improvement removes host
preparation work. It does not eliminate native arithmetic or device transfers.

Both wider-process simulator reports have the same:

| Counter | Both runs |
|---|---:|
| Commands / native loops / fences | 9,329,686 / 753,192 / 181,785 |
| Mesh tiles / busy cycles | 163,880,960 / 2,622,095,360 |
| VPU commands | 42,754 |
| Load bytes / Gets | 37,493,680,896 / 614,509,232 |
| Scale bytes | 1,155,792,896 |
| Store reads / Puts | 25,860,912 / 25,860,912 |

Host tracking remains off. These process counters are wider than the session
timer. Host stalls are 5,087,525,152 → 5,087,497,498; other memory observations
are retained verbatim rather than assumed equal.

The image grows **2,439,193,888 → 2,439,205,960 bytes** (12,072 more). Common
`.text` grows by 364 bytes and decoder catalog `.text` by 6,078 bytes. Static
BSS, packed weights, native workspace and 2 GiB runtime arena are unchanged.
Literal allocation sums remain 1,173,459,234 bytes/1,111 sites for prefill and
413,326,750 bytes/1,087 sites for decode, with no dynamic-sized sites. These sums
are not peak memory. This frozen image predates the newly implemented optional
cross-stage constant pool; no pool image saving is attributed to this run.

## Verification and reproduction

Run the standalone Python/NumPy reader:

```sh
python docs/evidence/cached_rhs_session_20261019/verify.py
# Also reopen both actual large images, section tables and candidate stage objects:
python docs/evidence/cached_rhs_session_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

Both modes passed. The reader reparses the complete ordered protocol, inclusive
timing arithmetic, all state hashes, every output byte and FP32 diagnostic. It
checks original source/precision/checkpoint/profile/settings, the same guarded
host math, actual original library suppliers, source-bound 44-call selection,
private endpoint contracts, unchanged original native schedules and resources.

The new archive retains **960 files in 4,189,860 compressed bytes**. It reuses
the exact candidate side of
[cached_latest_session_20261019](../cached_latest_session_20261019/README.md),
which pins the original cached archive/capture. Every recursive manifest,
archive and member is rechecked before namespace mapping. The unchanged prefill
is reused only after complete stage-object identity checks. Candidate logs,
outputs, generated code/IR, job status/specs, original recipes, frozen linker
producer and selected `libm.a` are retained. Large checkpoint-bearing objects,
ELFs, checkpoints and full toolchains remain external; optional checks reopen
actual candidate objects and both images. This is not a hermetic rebuild or
independent hardware qualification.

Reproduce with the retained decoder job spec (`--reuse-batch-rhs`) and original
latest prefill build, followed by the retained session link/run command arrays
in fresh task-local directories. Retain the original capture, source revisions,
selected tools, exact host-math guard, full output dumps and timing settings.
After real process termination, run the retained cycle comparator and
`readers/check_complete.py` with the new paths. A current compiler rebuild is a
new experiment; original frozen producer hashes remain authoritative.
