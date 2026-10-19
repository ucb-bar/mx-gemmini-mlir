# Consume compact source matrices instead of expanded repeated batches

The preceding RHS-reuse pass avoids repeated packing but keeps the expanded
source tensor. This optional extension feeds the adapter one matrix per proved
group. Merlin builds ordinary static views and, when required, a transpose of
the compact matrices. Upstream lowering can remove the dead expanded copies.
The target adapter reads RHS matrix `batch / repeat` and retains its private
packing within the synchronous batch call.

The proof follows exact source coordinates through pure copies, singleton
broadcasts, linear reshapes and matrix-axis transposes. It records transpose
orientation explicitly, including square matrices. Other consumers retain
their original tensors; compaction does not grant source movement, aliases or
asynchronous completion. The provider separately qualifies the changed read
ABI, original arithmetic/rounding, partial refusal and private workspace. The
flag `--compact-batch-rhs` requires `--reuse-batch-rhs`, integer-group packing
and explicit numerical policy. All flags remain experimental and default off.
Memory and performance benefits depend on the actual remaining source uses;
these measurements qualify the tested private cases rather than arbitrary fanout.

| Complete captured graph | Retain packing, expanded RHS | Compact RHS | Fewer cycles |
|---|---:|---:|---:|
| B6/M19/N23/K37, repeat3, transposed RHS | 765,994 | 731,571 | 4.4939% |
| B32/M1/N160/K64, repeat8, transpose and implicit zero rows | 5,504,459 | 2,368,224 | 56.9763% |
| Equal materialized RHS samples without source proof | 1,091,675 | 1,091,675 | 0% |

Every complete output word matches: 2,622 words for each tail/fallback graph,
5,120 for each one-token attention graph. Device work and transfer counters
match within every pair. For the tail graph these are 78 commands, 6 loops,
246 fences, 96 mesh tiles, no VPU commands, 24,576 load bytes and 768 scale
bytes. Counters cover the wider process with host tracking off.

Tail literal allocations sum to 51,568 → 17,454 bytes across 8 → 7 sites;
images are 57,184 → 57,200 bytes. Attention literal allocations sum to
2,642,152 → 184,478 bytes across 8 → 7 sites; images are 205,560 → 201,480
bytes. These are literal allocation sums, not peak-memory measurements.
Native private workspace is unchanged. The fallback selects zero compact
calls, preserves the original IR hash, all outputs, counters, allocations
and both 66,352-byte images despite equal captured RHS values.

Seven native comparisons also check the compact input ABI directly. Both FP4
and FP8 cover simultaneous M/N/K tails. Three invocations refresh changed first
and interior groups and match independent complete integer-product goldens.
LHS NaN, RHS NaN and finite-F32-to-BF16 overflow preserve every partial output
word, exterior guards and the actual explicit breakpoint. The verifier checks
the smaller RHS array extent directly in each retained ELF symbol table.
Cold native QK/PV adapter cycles are 15,111,468 → 2,275,452 and
15,376,044 → 2,537,704 relative to the original repeated-packing adapter. Those
native comparisons include both packing reuse and the compact ABI; the complete
graph pairs above isolate compaction after packing reuse.

The real forward and decoder IR each admit 44 compact calls across two shared
declarations, producing 22 RHS inputs of shape4×64×160 and 22 of shape4×160×64.
Those diagnostics are structural checks, not compiled-provider or timing
qualification. A fresh full forward is running in
`onepass160_rms_compact_batch_rhs`; it must pass the complete logit and original
source-relative gates before any whole-model improvement is attributed here.
No compact full-model speedup or cached-session result is claimed yet.

Generic source mechanism/tests are committed in Merlin `e3e951843`; target
ABI, encoding and integration in MX `dd6dfb2`. Fifty generic repetition/
broadcast tests pass, as do 111 focused target integration tests and the final
28-case RHS suite including two additional compact shared-call cases. Core
Ruff, package-doc generation, structure and docs gates pass. Normal mode
retains the earlier complete graph's exact 765,994-cycle result.

```sh
python docs/evidence/compact_batch_rhs_20261019/verify.py
```

The standalone stdlib/NumPy verifier reopens 843 retained files, 20 actual ELFs
and 110,932 observed output words. It checks payloads, independent native
goldens and partial outputs, guards, actual breakpoint instructions, compact
array extents, admissions, recipes, source/object identities, complete graph
words, original timers, resources, counters and fallback. The 935,760-byte
archive includes source/capture snapshots, generated IR/C, actual objects and
images, logs, commands and structural full-model diagnostics. It uses the
pinned adjacent broadcast ELF reader without importing compiler modules.
Toolchain executables, native startup dependencies and the simulator extension
remain external; this is not a hermetic rebuild or instruction-level proof.

Reproduce the native cases with the prior
[RHS-reuse qualifier](../repeated_batch_rhs_20261019/README.md), adding
`--compact-rhs`. For the complete graph, capture groups2/repeat3/M19/N23/K37
with `--transpose-rhs`; build both with the same policy, integer-group packing,
prepacking, `--reuse-batch-rhs` and structural admission, adding
`--compact-batch-rhs` only to the candidate. The attention case uses
groups4/repeat8/M1/N160/K64 plus `--virtual-zero-padding`. The fallback capture
adds `--materialized-rhs`. Exact original stage command arrays are retained
under `graphs/*/commands/`; select fresh task-local output paths.
