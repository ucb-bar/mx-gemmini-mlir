# Compact RHS inputs in a complete TinyLlama forward

Removing private expanded RHS materialization improves the complete 160-token,
22-layer token-ID-to-logits forward by another **1.01982%**. Both runs complete
on the pinned simulator and preserve all 5,120,000 F32 logit words exactly.

| Completed execution | Modeled cycles |
|---|---:|
| Ordinary repeated-RHS packing reuse | 10,433,218,556 |
| Compact source RHS plus packing reuse | 10,326,818,367 |
| Independent reference source, same logical boundary | 12,762,360,525 |

The candidate is **19.0838% below the reference implementation’s cycle count**, and 35.3954% below
the corrected compiler's 15,984,649,026 cycles. The inaccurate older compiler
archive is not the control. The unchanged source-relative gate passes on 33
retained full source distributions: mean KL 0.0637956270, worst-row KL
0.1699797580 and 27/33 top agreement. Compiler FP32 top agreement remains 133/160.
The complete output SHA-256 remains
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
No complete independent 160-row source distributions or source bit identity are
claimed.

Merlin derives one source matrix per repeated group, including a compact
transpose when required, and changes only qualified consumer inputs. The target
endpoint reads compact matrix `batch / repeat` while retaining its original
packing order, private workspace, numerical behavior and synchronous completion.
All 44 selected calls have batch 32, repeat eight and four compact input matrices.
QK and PV uses must be considered together when proving that shared expanded
copies can disappear. This is source graph information unavailable to a matrix
kernel receiving an already expanded opaque tensor.

This frozen full-forward image was compiled **before the later privacy cost
guard**. It does not implement that guard or the newer separately retained
ordinary fallback ABI. A fresh reanalysis of its exact pre-rewrite source using
the current generic guard proves that all 44 selected expanded-input uses are
private. The result therefore qualifies this private workload; it does not
qualify unguarded compaction when expanded outputs escape. The measured
live-output regression and current guarded fallback remain in
[compact_batch_cost_guard_20261019](../compact_batch_cost_guard_20261019/README.md).
The fresh guarded compact cached-session image is a separate running experiment.

The compiler's original 157 kernel schedules, precision, immutable packed weight
bytes, workspace extents, quantizer object, host math policy, captured inputs,
simulator settings and logical uninstrumented timing boundary match. Only the
two ordinary repeated-input native helpers are replaced by compact-input
helpers, alongside the corresponding ABI/source rewrites. This measures the
complete transformation, including compact view preparation and remaining host
execution.

Device work and transfer counters match: 3,312,204 commands, 38,608 loops,
492,001 fences, 40,970,240 mesh tiles, 655,523,840 mesh busy cycles, 29,364 VPU
commands, 3,939,506,496 load bytes and 107,198,464 scale bytes. Loads use
85,815,700 Gets; store reads and Puts are both 17,154,476. Host tracking is off.
Host-stall observations decrease by 23,395 cycles and events by 7,180, so not
every wider-process counter is treated as identical. These counters have a
wider boundary than the stage timer. The gain removes host materialization;
device DMA is not eliminated.

Literal allocation sizes sum to **1,343,648,814 → 1,288,269,192 bytes**, a
55,379,622-byte reduction, across 1,061 → 1,038 sites with no dynamic-sized
sites. These are not peak-memory measurements. Both retain a 2 GiB arena at
`0xd0000000` and the same 10,579,968-byte native workspace. The image shrinks
from 1,330,849,968 to 1,330,788,912 bytes (61,056 fewer); allocated `.text`
shrinks by 63,238 bytes. Allocated rodata, data, BSS and checkpoint sections are
unchanged. Original fallback kernels remain linked; no smaller reserved arena
or native workspace is claimed.

Standalone verification needs Python and NumPy:

```sh
python docs/evidence/matched_compact_batch_rhs_forward_20261019/verify.py
```

The reader independently checks both full word streams, the original source
boundary and quality gates, capture identities, counters, receipts, completed
process observations, source rewrite records, original kernel identities,
native endpoint contracts, resource observations and current privacy-reanalysis
receipt. The optional checks reopen both actual large images and section tables,
and rerun the generic source privacy analysis using the recorded compiler:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
  python docs/evidence/matched_compact_batch_rhs_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018" --check-source-privacy
```

The complete reader, actual image rechecks and all 44 source privacy rechecks
passed. Structural admission establishes image/declared-ISA custody rather
than independent hardware or instruction-policy qualification. Source reanalysis
does not insert a guard into the historical executable.

The new archive retains 901 files in 10,014,560 compressed bytes. Supplemental
privacy observations and the exact generic proof source are separately hashed.
It reuses the baseline side from
[matched_batch_rhs_forward_20261019](../matched_batch_rhs_forward_20261019/README.md),
which in turn reuses the preceding reverse-transport archive. Every dependency
manifest/archive and every member is rechecked before use; baseline records are
not reconstructed from mutable sources. Candidate logs, generated IR/native
sources/schedules/headers, commands/status, frozen small objects, comparison
and actual resource observations are retained. Large ELFs, checkpoints and
toolchains remain external; this is not a hermetic rebuild.

For reproduction, replay each retained original job command in a fresh task-local
directory using its original selected source revisions and pinned inputs/tools.
The candidate adds `--compact-batch-rhs` to the ordinary RHS-reuse build. Validate
complete outputs only after actual process termination, then run the original
matched full-forward comparison with the independent source and token-lookup
boundary archives. A current compiler rebuild includes the later guard and is
a new experiment, not reproduction of this frozen image hash.
