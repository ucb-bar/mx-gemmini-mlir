# Resident matrix epilogue: two candidates rejected

Keeping a matrix output resident is not automatically profitable or measurable
on the selected execution provider. This probe retains the original route:
one candidate fails timing-work qualification; the other saves load bytes but
is **25.1209% slower** in the measured small section. Neither is enabled in a
production compiler pass.

Three independent admitted executables evaluate a 32×32×64 E4M3 matrix product,
partitioned into two K=32 native waves, followed by multiplication by the finite
BF16 scalar `0x3f8d` (1.1015625). Each executable receives the same three distinct
integer ±1 input pairs. All 3,072 BF16 output words per executable match an
independent integer matrix product followed by the required BF16 rounding and
scalar rounding. All output/matrix guard words remain intact. The images run
separately; a baseline execution does not warm a candidate.

| Route | Cycles for the three calls | Timing decision |
|---|---:|---|
| Native matrix to DRAM, reload, scalar VPU, readout | 1,624 / 1,463 / 1,463 | Control; 4,550 total |
| Native matrix without C store, store-only loop, resident scalar VPU | Not accepted | Work counters incomplete |
| Explicit scratchpad matrix schedule, resident scalar VPU | 2,005 / 1,844 / 1,844 | 5,693 total; reject regression |

The timer includes issuer dispatch, configuration, transfers, matrix work, VPU
work and completion fences. It excludes encoded operand preparation, guard
initialization, console output and validation. This is an **encoded native
section**, not a full-model result or the original high-level broadcast kernel.

## Native store-only candidate

The exact pinned source shows that the scratchpad flag selects the raw loop
handler before native DRAM operand loading. Consequently the candidate preserves
both original native compute waves, suppresses their C-to-DRAM store, waits,
then requests an accumulating K=0 loop with A/B/D/EX skipped and scratchpad
output enabled (`rs2=0x278`). K=0 avoids repeating arithmetic in the retrieved
functional raw-loop implementation. It then issues the same in-place BF16 VPU
scalar operation and readout as the control.

Although all three functional outputs are correct, the timing report accounts
for only **16 mesh tiles, 256 mesh busy cycles, one VPU command and 128 scale
bytes**, versus the expected **48 / 768 / three / 384**. It reports five loops
instead of the nine requested native/store-only loops. The apparently low
timers are therefore **refused as performance evidence**. Correct functional
results and process exit zero do not establish completion of modeled device
work. This observation does not establish the root cause in the provider, or
prove equivalence to RTL execution.

## Explicit scratchpad candidate

The second candidate uses the existing physical scratchpad scheduler, preserving
the two K=32 arithmetic partitions. It loads operands/scales explicitly, retains
the final BF16 matrix tile in its planned scratchpad region, applies the scalar
VPU operation there and reads out once. Its mesh/VPU/scale work counters are
complete and match the control.

| Wider-process counter, all three calls | Control | Explicit resident |
|---|---:|---:|
| Commands | 120 | 141 |
| Loops | 6 | 6 |
| Fences | 410 | 425 |
| Mesh tiles / busy cycles | 48 / 768 | 48 / 768 |
| VPU commands | 3 | 3 |
| Load bytes | 18,432 | 12,288 |
| Scale bytes | 384 | 384 |

These process counters include runtime/console activity and are wider than the
section timer. Host memory tracking is off. The additional explicit commands and
fences outweigh the saved transfers in this case. The resident image is 38,112
bytes versus 37,664 for the control; no peak-memory measurement is claimed.
This counterexample rejects broad replacement with the current manual route.
Other shapes, larger connected operations and improved scheduling remain
unmeasured; the result does not prove that all resident epilogues are slower.

## Scope and reproduction

This experiment covers finite integer inputs, a finite scalar, one output tile,
two K waves, changed inputs, exact words and canaries. It does not qualify
general exceptional-value behavior, F32 NaN-refusal preservation, partial
publication on refusal, model-wide ownership or an asynchronous schedule.
Ordinary F32 consumers still require their existing numerical contract; a
generic rewrite must additionally prove layouts, capacity, lifetime, aliasing
and completion. No opcode, profile, simulator binary or admission check changed.

The recovered Gemmini ISA bytes match the registered profile pin. The archive
also retains the exact retrieved RTL loop and registered libgemmini-submodule
sources with URLs/digests. Retrieval alone does not establish that those sources
rebuild the existing extension byte-for-byte. The actual executions use the
unchanged selected extension SHA
`5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5`.
The small images pass structural admission, whose scope is declared ISA/digest
custody rather than an instruction policy or independent hardware qualification.

Run the standalone reader with Python and NumPy:

```sh
python docs/evidence/native_resident_epilogue_rejections_20261019/verify.py
```

It hashes all 107 archived files (95,652 compressed bytes), reopens all three
actual ELF byte streams and their admission reports, independently derives the
full numerical oracles, reparses every output word/counter and enforces both
rejections. It also checks the exact source-study custody and store-only
mutation. Complete verification passed.

For a fresh native execution, use the archived
`probe/probe_native_resident_epilogue_20261019.py` with the recorded compiler
sources/toolchain/startup and selected extension. The source expects the MX
repository as its working directory and task-local captures under
`$TMPDIR/mx_model_goal_20261018`. For example, in the retained workspace:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
MERLIN_READELF=/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-readelf \
MX_EPILOGUE_PROBE_DIR=native_resident_epilogue_fresh \
  "$TMPDIR/smolvla-venv/bin/python" \
  "$TMPDIR/mx_model_goal_20261018/probe_native_resident_epilogue_20261019.py"
```

The first attempt stopped before simulation because the readelf environment was
missing; its original failure log is retained. The fresh completed experiment
sets that existing tool explicitly and preserves admission. Generated C, payload,
arrays, schedules, objects, all three ELFs, commands, original runner and logs
are retained. Toolchains/startup source and the extension remain external; this
is not a hermetic rebuild or a whole-model optimization result.
