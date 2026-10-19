# Complete recurrent fixture and full pooled-image admission

The production session linker now completes its full observer checks after the
allocator object is produced. A real recurrent-session regression passes link,
admission, execution, state handoffs, all eight output/state hashes and all four
complete output tensors. The corrected full TinyLlama pooled image also passes
admission and actual image/section checks; its complete simulation is running.
These are distinct qualifications.

## Completed small session

The same previously qualified compiled stages execute one prefill and three
recurrent decodes. Every one of the 2,048 F32 output words matches the original
session, as do all eight output/state hashes.

| Inclusive interval | Original | Shared encoded constants |
|---|---:|---:|
| Reset | 1,330 | 1,330 |
| Prefill plus preparation | 160,739 | 160,739 |
| Three decodes plus state handoff | 478,159 | 478,065 |
| Total | 640,228 | 640,134 |

Timing boundaries, original stage objects, captured inputs/state streams,
checkpoint pool, profile, simulator, exact math guard and library are unchanged.
This small deterministic fixture is regression coverage, not a full-model
speedup claim or a new numerical acceptance threshold.

Encoded storage decreases 2,176 → 1,088 bytes, but the actual ELF grows
**111,848 → 128,256 bytes**. The current catalog/pool placement has additional
file-layout overhead. Reduced constant payload does not promise a smaller
executable for every shape. The feature remains explicitly selected and off by
default; small-shape resource profitability is not inferred from payload alone.

The first full-model attempt stopped before simulation because the complete
observer check ran before `session_malloc.o` existed. Its actual terminal status
and original error are retained. The fix moves the check after compilation and
still inspects every nonconstant input. The added production-link regression
would fail on that ordering bug; it completed successfully in 3.07 seconds.

## Admitted large image; execution pending

The corrected fresh link uses the exact original latest prefill/decode objects
and 128+32 capture. It reconstructs both original catalogs byte-for-byte,
regenerates/recompiles all **625 selected reader/issuer objects per stage** to
the original machine bytes, closes all constant uses and verifies the actual
pooled constant object. The generic pool contains 310 unique code/scale blobs
for 620 original bindings.

| Resource | Original large image | Pooled large image |
|---|---:|---:|
| Packed constant payload | 2,133,491,712 bytes | 1,066,745,856 bytes |
| Actual ELF | 2,439,193,888 bytes | 1,373,933,592 bytes |
| Reserved runtime arena | 2 GiB | 2 GiB |
| Simulator memory span | 7,516,192,768 bytes | 7,516,192,768 bytes |

The measured ELF reduction is **1,065,260,296 bytes**. The arena relocates from
`0x110000000` to `0xd0000000` after recomputing the static extent; the original
captured weight blob remains at `0x200000000`. Full allocated section tables are
retained. These are static/link observations, not peak-memory measurements.

The actual pooled image SHA-256 is
`5ce98c74ad5f6df9fc4d6c5055bc8ca8f4cbf50de68b2980989758652a1cb4e9`.
Its fresh selected math-link recipe, source/tool pins, complete constant-reader
proof, structural admission and observed live simulator settings are retained.
The original captured weight pool, precision, source stage objects, exact math
guard, memory span and logical timing boundary match. The runtime uses the same
selected extension; no simulator or hardware profile was modified.

**Complete large-session output, numerical and cycle qualification remains
pending.** The admitted image is not a completed TinyLlama performance result.
It is separate from the completed repeated-RHS cached session and does not
include that newer decoder transformation. Structural admission establishes
digest/declared-ISA custody, not independent hardware qualification.

## Verification

```sh
python docs/evidence/shared_catalog_constants_session_20261019/verify.py
# Also reopen both actual large ELFs and allocated section tables:
python docs/evidence/shared_catalog_constants_session_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

Both modes passed. The standard-library reader reparses the complete small
session protocol, timings, hashes and full bytes; checks actual small ELFs,
admissions and original recipes; and checks all 1,250 large reader source/object
proofs against the pinned original source archive. It reopens both actual large
images and section tables when requested. Large live-state snapshots are
explicitly nonterminal.

The archive retains **2,655 files in 649,748 compressed bytes**. It includes
small complete images/runs, all fresh selected large reader sources/objects,
pool bindings/proofs, link/admission/recipe observations, source snapshots and
reproduction commands. Its original large-control dependency is
[cached_latest_session_20261019](../cached_latest_session_20261019/README.md);
every retained member and dependency ledger/hash is checked. Large images,
checkpoint-bearing objects, raw model payloads and full toolchains remain
external. This is not a hermetic rebuild.

For the complete small regression, set `MERLIN_TEST_POOL_SESSION_FIXTURE` to the
previously qualified guarded fixture and run
`tests/test_catalog_constant_pool.py::test_complete_session_materializes_allocator_before_observer_admission`
with a fresh task-local pytest directory and the original compiler environment.
Exact small link/run/comparison commands are retained at the archive root. For
the large run, replay `large/job/link.command.json` followed by
`large/job/run.command.json` in fresh directories with the original inputs/tools.
Wait for real process termination, then compare every complete output and the
unchanged inclusive timing boundary; do not infer completion from this archive.
