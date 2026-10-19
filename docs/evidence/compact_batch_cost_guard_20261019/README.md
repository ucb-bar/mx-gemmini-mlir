# Reject compaction when expanded tensors remain observable

A source-proved repetition does not by itself make compaction profitable.
Returning the expanded RHS keeps its original materialization live. Adding a
compact transpose then increases work, despite preserving both outputs.

| Complete graph | Cycles | Literal allocation sum | Image bytes |
|---|---:|---:|---:|
| Live expanded RHS, ordinary packing reuse | 778,849 | 51,568 | 57,336 |
| Live expanded RHS, unguarded compaction | 782,715 | 58,450 | 57,360 |
| Live expanded RHS, guarded ordinary fallback | 778,849 | 51,568 | 57,592 |
| Private expanded RHS, ordinary packing reuse | 765,994 | 51,568 | 57,184 |
| Private expanded RHS, guarded compaction | 731,571 | 17,454 | 57,432 |

The guard rejects the measured 0.4964% regression and preserves the private
case's 4.4939% improvement. Allocation sums are not peak-memory measurements.
The unused qualified alternative adds a small image-size cost. Device work and
transfer counters match within each workload: 78 commands, 6 loops, 96 mesh
tiles, no VPU commands, 24,576 load bytes and 768 scale bytes. The live-output
workload has 248 wider process fences; the private workload has 246. Host
tracking is off. Their distinct observation boundaries are not compared as one
workload.

Merlin's generic `private_repeated_batch_inputs` checks the complete selected
consumer/copy/view graph. It repeatedly removes candidates with escaping
intermediates until the remaining graph closes. Public source values before
the broadcast remain legal. Tests cover shared consumers, sibling views,
public expanded/source outputs, changed proofs and fixed-point propagation.
The check mutates no IR and grants no numerical, alias or scheduling permission.

MX compiles both qualified input ABIs. Private candidates use the compact ABI;
other proved repetitions retain ordinary packing reuse. Both endpoints keep
the original numeric, lifetime, synchronous-completion and workspace contracts.
The ordinary fallback's object/source metadata is revalidated even when the
private case can compact. This is a structural cost guard backed by measured
cases, not a general fusion cost model. Default optimization flags remain off.

Every first-output word is dumped and compared: 2,622 F32 words per execution.
The live RHS has 5,106 F32 words. Its complete-buffer hash matches independently
reconstructed full bytes from the captured input and exact repeat/transpose
coordinates. Both additional-output references are mandatory: the comparison
tool refuses missing or changed references before writing a report. The archive
retains actual negative CLI invocations, logs and the changed full-size reference.
Additional outputs are identified as complete reference/hash observations;
they are not described as additional console word dumps.

Both the real forward and decoder source graphs retain all 44 private compact
candidates under this guard. Those are source-use diagnostics, not new compiled
or timing results. The running full forward uses the earlier frozen compact
image; a fresh cached decoder with this guard selects all 44 compact calls and
is being evaluated through its complete session. No new whole-model speedup is
claimed from these fixture measurements.

```sh
python docs/evidence/compact_batch_cost_guard_20261019/verify.py
```

The standalone stdlib/NumPy verifier reopens 339 files, five actual ELFs,
13,110 first-output words and three complete live-RHS hash observations. It
checks original captures, independent RHS bytes, actual words/hashes, counters,
timings, resources, admissions, recipes, compiled source/object identities,
guard selections and negative-reference checks. It imports only the pinned
adjacent broadcast ELF reader. The 340,548-byte archive retains all five stage
builds, commands, logs, captures and source/test snapshots. Toolchains and the
simulator extension remain external; this is not a hermetic rebuild or an
independent whole-model numerical qualification.

Source commits are Merlin `a05c2f807` and MX `47a82f8`. Fifty-seven generic tests
and 120 target integration tests pass, with core Ruff/package-doc/structure/docs
and target Ruff checks. The earlier unguarded source is preserved in the frozen
negative stage artifacts; historical records were not amended.

Reproduce the preceding
[compact graph qualifier](../compact_batch_rhs_20261019/README.md), adding
`--return-rhs` to the capture tool. Build a packing-reuse control and a
`--compact-batch-rhs` candidate with current sources. The candidate must report
zero compact calls and one ordinary-reuse fallback. For multiple-output
comparison, pass `--additional-output-reference 1=rhs.bin`, containing complete
independently expanded/transposed RHS bytes. Exact stage and negative command
arrays are retained under `commands/` and `negative_checks/`.
