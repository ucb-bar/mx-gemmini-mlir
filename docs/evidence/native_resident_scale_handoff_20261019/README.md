# Native matrix → scalar VPU: explicit scale-state handoff

A native matrix with suppressed DRAM C storage, followed by a store-only
scratchpad loop and scalar BF16 VPU multiply, now completes its timing work
across repeated calls. Two unmanaged scale configurations after readout release
both previously managed halves before the next native invocation. The sequence
retains the native computation and original BF16 result rounding. This is a
finite-integer native feasibility experiment; no production resident rewrite,
full-model resident gain, exceptional-value policy or hardware qualification is
established.

The original store-only experiment and the two new nonzero-K alternatives
produce correct functional words but leave subsequent modeled work incomplete.
The actual console says `gemmini perf: MODEL DEADLOCK`; Spike still exits zero.
Nonzero K with EX skipped, and explicitly executed zero-product work, both fail
the timing gate. Changing K alone does not fix the transition. The new sequence
uses K=0 with LOAD_A/B/D and EX skipped, then copies the retained accumulator to
scratchpad. The pinned RTL marks those stages complete directly from the skip
bits; this source study does not establish binary-to-source equivalence or an
RTL execution result.

Every table row sums three invocations with different ±1 matrix operands, E8M0
scales of 127 and BF16 scalar `0x3f8d`. Each side is an independently admitted
ELF. All output words match the independent integer-dot/BF16 oracle and each
other, and all exterior guards remain intact. Timers include configuration,
native matrix work, the epilogue, transfers and completion; output checking and
printing follow the timer. These counts are modeled Spike cycles.

| Shape M×N×K | Baseline cycles | Resident cycles | Fewer cycles | Load bytes, baseline → resident |
|---|---:|---:|---:|---:|
| 16×16×32 | 2,052 | 1,796 | 12.4756% | 4,608 → 3,072 |
| 16×64×64 | 4,625 | 4,445 | 3.8919% | 21,504 → 15,360 |
| 32×32×64 | 4,550 | 4,403 | 3.2308% | 18,432 → 12,288 |
| 64×16×256 | 13,717 | 13,196 | 3.7982% | 67,584 → 61,440 |
| 64×64×96 | 14,799 | 14,112 | 4.6422% | 61,440 → 36,864 |

The first 32×32×64 release probe used an additional fence between the two scale
configurations: 4,550 → 4,430 cycles. The subsequent sweep queues both configs
in order and drains once. Complete mesh tiles/busy cycles, three VPU executions
and scale bytes match in every accepted pair. Resident execution adds three
store-only loops. Commands change respectively 60→63, 120→105, 120→105, 264→249
and 288→201; wider process fences increase by three. Load bytes decrease by
exactly `3*M*N*2`, removing the C reload. This is not a claim of zero DRAM
traffic. Both driver buffers remain allocated. ELF sizes change respectively
23,648→23,648, 40,928→40,992, 37,664→37,728, 88,416→88,416 and 82,400→82,016 bytes.

The archive also preserves actual failed harness attempts. The first release
script failed its final report assembly on a removed role key after execution.
The initial shape sweep retained a fixed output count. Its first repair then
changed a payload offset while replacing a matching buffer-size literal; the
16×16 first matrix consequently used the wrong scale pointer. The corrected
script substitutes only array/loop bounds. The independent reader reconstructs
every input payload and checks every generated driver call offset, including
the earlier successful larger shapes. These harness failures do not establish
an accelerator numerical failure.

MX commit `92c2b93` makes stage/session observers, matched full-forward/session
comparators and native tile qualifiers reject the actual deadlock diagnostic,
even with receipt-bound exit-zero functional outputs. Fifty-eight focused tests
pass, including prefix/suffix/interleaved diagnostics and source/session timing
receipts. The separate audit reopened all 91 retained completed stage logs and
22 completed timing-session logs at this checkpoint; none contains the marker.
This preserves existing reported results. Marker absence alone is not proof of
complete modeled arithmetic work; the native probes additionally require their
expected arithmetic and transfer counters.

Verify with Python and NumPy:

```sh
python docs/evidence/native_resident_scale_handoff_20261019/verify.py
python docs/evidence/native_resident_scale_handoff_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The default reader checks all archive members, full output words, input payloads
and pointers, guards, admitted actual ELFs, native source changes, complete-work
counters and actual timing-model rejections. It also checks the frozen completed
log audit and receipt bytes. `--build-root` additionally reopens and hashes every
audited actual log and receipt; both verification modes passed. The reader does
not independently replay the timing model. The archive contains 912 files in
268,080 compressed bytes, including the executed images/objects, generated C,
MK, payloads, invocation/settings, rejected probes, public source snapshots,
producer/test snapshots and startup sources. Toolchain/transitive headers remain
external, so this is not a hermetic rebuild.

Reproduce from the MX directory using the revisions and tool paths recorded in
`metadata.json`, a Python environment with NumPy/xDSL/Merlin, and fresh task-local
output directories:

```sh
python -m tarfile -e docs/evidence/native_resident_scale_handoff_20261019/probes.tar.xz \
  "$TMPDIR/resident-handoff-inputs"
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export MERLIN_MLIR_INSTALL=/path/to/llvm-install
export MERLIN_READELF="$MERLIN_MLIR_INSTALL/bin/llvm-readelf"
MX_EPILOGUE_PROBE_MNK=32,32,64 \
MX_EPILOGUE_PROBE_DIR=native_resident_replay_new \
python "$TMPDIR/resident-handoff-inputs/cases/m32n32k64/probe_native_resident_shape_sweep_20261019.py"
```

The retained script uses the recorded RISC-V/startup paths and selected profile;
other tested shapes use the table's extents. Its original frozen source is
retained beside each run. Rebuilding with changed sources/tools is a new
experiment. A general compiler endpoint still needs explicit layout/capacity,
private-use/lifetime/completion contracts, broader values/precisions/tails and
multi-output-tile qualification, then a complete-model comparison.
