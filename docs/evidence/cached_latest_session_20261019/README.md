# Complete TinyLlama cached session with newer generic passes

The completed 128-token prefill plus 32 one-token cached decodes improves from
**55,871,481,378 to 41,567,160,094 modeled cycles**, a **25.6022% reduction**
against the earlier optimized compiler session. Both real processes terminated
with exit code zero. All 99 ordered output/state hashes and all 33 complete
32,000-logit tensors are identical. This preserves the compiler control's
outputs; no independent reference cached baseline or generation-quality acceptance
threshold has been established.

| Included interval | Earlier optimized control | Newer passes | Reduction |
|---|---:|---:|---:|
| Reset | 4,505,788 | 4,505,788 | 0% |
| Prefill, including preparation | 11,539,855,800 | 8,885,018,358 | 23.0058% |
| 32 decodes, including state handoff | 44,327,119,790 | 32,677,635,948 | 26.2807% |
| **Total** | **55,871,481,378** | **41,567,160,094** | **25.6022%** |

The original corrected cached compiler used 74,065,304,284 cycles, so the newer
session is 43.8777% below that control. Its complete observations remain in
[cached_session_reuse_20261019](../cached_session_reuse_20261019/README.md).
These are separate cached-session measurements, not the independent 160-token
forward or a stitched-kernel issuer swap.

The unchanged timing boundary includes reset, program state handoff and the
existing measured step intervals. Console output and phase bookkeeping remain
outside the timer. Preparation costs are 22 prefill cycles and 4,506,555 decode
cycles in both images. Actual raw step totals are 55,862,469,013 and
41,558,147,729 cycles. Inclusive instructions decrease from 50,772,517,196 to
36,479,634,801. The 14,304,321,284-cycle saving equals the 14,292,882,395-instruction
reduction plus the observed 11,438,889-cycle decrease in host stalls.

The stages use the same captured inputs, checkpoint, placement policy, numerical
precision decisions, target profile, shared checkpoint pool, simulator,
extension, memory size and host-tracking setting. Newer passes include connected
broadcast pointwise work, broadcast operand preparation, integer word-group
packing, guarded host math, and private BF16 transport. The prefill selects
44 forward and 44 reverse matrix/VPU transport edges; decode selects 44 forward
edges and no reverse edges. Each stage selects 45 connected multiply pairs and
22 sigmoid/multiply chains. Both retain the **legacy native tile policy**.
This experiment does not include repeated-RHS reuse or compact-RHS inputs; those
are separate still-running complete sessions.

The original generated link selected a global guarded `expf` wrapper. The archive
retains both attempted link recipes, the selected second recipe, its exact
historical Python producer, the wrapper source/object and the selected `libm.a`.
The reader verifies final ELF binding, source/object/guard identities and the
linker's actual definitions of `expf` and `__ieee754_expf`. The reference had no
such wrapper. The guard and fallback obligations are preserved.

## Numerical evidence

Every complete published output file is checked against its console HEX stream
and recorded SHA-256: 128,000 bytes per tensor, **1,056,000 F32 values** across
the 33 steps. Ordered hashes also cover the KV state and position output after
each step. All values match the earlier completed compiler session exactly.

Recomputed diagnostics against the captured FP32 references are unchanged:
mean KL 0.07095571052388557, worst-row KL 0.18586350146524108, minimum cosine
0.9724467380582315, and 28/33 top-token agreement. These diagnostics do not
establish a new acceptance gate or independent source equivalence. No separate
fast-run observation is invented or inferred from these two different compiled
images. The receipt status remains `complete_numeric_qualification_pending`.

## Traffic and resources

The simulator counters cover the wider process with `mem.host_tracking=0`;
their scope is wider than the measured session intervals.

| Counter | Earlier control | Newer passes |
|---|---:|---:|
| Commands | 9,797,152 | 9,329,686 |
| Native loops | 753,192 | 753,192 |
| Fences | 182,241 | 181,785 |
| Mesh tiles | 163,880,960 | 163,880,960 |
| Mesh busy cycles | 2,622,095,360 | 2,622,095,360 |
| VPU commands | 42,730 | 42,754 |
| Load bytes | 37,556,271,872 | 37,493,680,896 |
| Scale bytes | 1,155,792,896 | 1,155,792,896 |
| Gets | 618,421,168 | 614,509,232 |
| Store reads and Puts, each | 29,781,040 | 25,860,912 |

The image grows from **2,433,905,848 to 2,439,193,888 bytes** (5,288,040 more).
Both reserve the same 2 GiB runtime arena. Native workspace stays at 8,464,384
bytes for prefill and 1,059,840 bytes for decode. Literal allocation-site sums
decrease from 1,307,158,302 to 1,173,459,234 bytes for prefill and from 414,196,634
to 413,326,750 bytes for decode. Each stage removes 67 allocation sites; neither
has dynamic-sized sites. These sums are **not peak-memory measurements**.
Allocated image sections are recorded separately in `resources.json`.
Static catalog BSS grows from 12,447,808 to 17,781,312 bytes for prefill and from
1,416,256 to 1,499,264 bytes for decode: **5,416,512 additional static bytes**
overall. Thus lower temporary allocation sums do not imply lower total or peak
memory. The native-workspace counts above exclude these additional helper
buffers.

## Verification and reproduction

The standalone reader needs Python and NumPy:

```sh
python docs/evidence/cached_latest_session_20261019/verify.py
```

To also reopen both actual multi-gigabyte images, their allocated section tables,
and the candidate's original stage objects:

```sh
PYTHONDONTWRITEBYTECODE=1 python \
  docs/evidence/cached_latest_session_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The archive reuses the exact `virtual` role and capture from the earlier cached
session archive. Its complete manifest/archive and every member are hashed
before namespace mapping. Duplicate new archive bodies are represented by an
explicit, checked alias ledger. The new archive contains 1,843 logical members,
1,680 unique bodies and 4,830,056 compressed bytes. The reader replays the complete ordered session
protocol, timing arithmetic, all hashes, full logits and FP32 diagnostics, and
checks source, object, resource, guard and original link-recipe custody.
The complete reader, both actual image/section rechecks and all candidate stage
object rechecks passed.

For a new execution, replay the retained prefill/decode job specs followed by
the retained session `link.command.json` and `run.command.json`, using fresh
task-local directories, the recorded frozen sources, selected tools and original
capture. After real process termination, run the retained
`compare_linked_session_cycles.py` against the earlier completed virtual session
and rerun `readers/check_complete.py` with the corresponding new paths.
The reader script retains the original task paths; adapt paths explicitly for
a new workspace. A current compiler rebuild produces a new experiment.

Large checkpoint-bearing catalog objects, ELFs, checkpoint tensors, large
generated headers and full toolchains remain external. Their recorded hashes
are distinguished from retained bytes; optional checks reopen the actual
candidate objects and both images. This is not a hermetic rebuild. Structural
admission establishes image/declared-ISA custody, not independent hardware
qualification. No thresholds or timing boundaries were changed.
