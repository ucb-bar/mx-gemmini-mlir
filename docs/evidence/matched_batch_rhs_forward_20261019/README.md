# Repeated RHS preparation in a complete TinyLlama forward

Source-proven repeated RHS packing reduces a complete 160-token, 22-layer
token-ID-to-logits forward by **5.22987%** relative to the preceding compiler
control. Both simulator processes exited zero, and every one of the 5,120,000
F32 output words is identical.

| Completed execution | Modeled cycles |
|---|---:|
| Preceding compiler, reverse BF16 transport | 11,008,973,639 |
| Additional repeated-RHS preparation reuse | 10,433,218,556 |
| Independent Nicolás source, same logical boundary | 12,762,360,525 |

The candidate uses **18.2501% fewer cycles than Nicolás** and 34.7298% fewer than
the corrected compiler's 15,984,649,026 cycles. The older 15,170,996,572-cycle
compiler archive has different numerical quality and is not this control.

The complete compiler outputs retain SHA-256
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
The unchanged source-relative gate passes on 33 archived full source
distributions: mean KL 0.0637956270, worst-row KL 0.1699797580, and 27/33 top-token
agreement. Compiler agreement with the FP32 top token remains 133/160. Full
independent source distributions exist for only 33 positions; gate acceptance
does not imply source bit identity or establish missing source rows.

Merlin proves equal adjacent RHS matrices from their actual source coordinates
through pure broadcasts and views. MX retains one packed RHS per proved group
inside a synchronous native batch call and refreshes it on every invocation.
The candidate selects 44 calls through two shared declarations and two additional
native kernels. Each batch of 32 matrices has repeat factor eight. This avoids
repeated host quantization and packing. The expanded source tensors and device
transfers remain; compact-input execution is a separate pending experiment.

Original arithmetic, native tiles, precision, checkpoint codes/scales, execution
policy, input IDs, profile, extension, simulator settings and uninstrumented
logical timing boundary match. Shared generated sources are unchanged except
for ABI shims and checkpoint assembly paths. Both use the identical frozen
runtime quantizer object. Kernel schedules and original workspace extents are
unchanged. This isolates the selected packing-reuse feature, including its
guards, code layout and complete host/device execution costs.

Wider process device counters match: 3,312,204 commands, 38,608 loops, 492,001
fences, 40,970,240 mesh tiles, 655,523,840 mesh busy cycles and 29,364 VPU commands.
Loads remain 3,939,506,496 bytes in 85,815,700 Gets; scale bytes remain 107,198,464.
Store reads and Puts are both 17,154,476. Host tracking is off. Host-stall
observations increase by 12,789 cycles and event counts by 12,143; not every
counter is assumed equal. These counters have a wider scope than the stage timer.

The ELF grows from 1,330,837,896 to 1,330,849,968 bytes (+12,072); allocated
`.text` grows by 6,122 bytes. Allocated rodata, data, BSS and checkpoint sections
are unchanged. Both reserve a 10,579,968-byte native workspace and a 2 GiB arena
at `0xd0000000`. Literal allocations sum to 1,343,648,814 bytes across 1,061
sites, with no dynamic-sized sites, in both builds. These are not peak-memory
measurements. Full section tables and resource observations are retained.

The standalone reader needs only Python and NumPy:

```sh
python docs/evidence/matched_batch_rhs_forward_20261019/verify.py
```

It checks both complete word streams, the original source boundary and quality
gates, actual counters, capture/input identities, completed process observations,
recipes, accepted structural admission receipts, live settings, frozen quantizer
sources/objects, host math selection, original kernel identities, 44 selected
source-call records and resources. It checks source/rewrite receipts; it does
not independently rerun compiler legality analysis or prove arbitrary input
equivalence. General source/native/tail/refusal qualifiers remain in
[repeated_batch_rhs_20261019](../repeated_batch_rhs_20261019/README.md).

Optional verification rehashes both actual retained images and independently
reads their allocated section tables:

```sh
python docs/evidence/matched_batch_rhs_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

Both the complete reader checks and actual image rechecks passed. Structural
admission establishes image/declared-ISA custody, not independent instruction
policy or hardware timing qualification.

`executions.tar.xz` adds 901 files in 9,820,528 compressed bytes: the candidate's
full log, receipts, input/output source IR, source proof records, generated native
sources/schedules/headers, frozen quantizer and host math objects, complete job
commands/status, comparison and actual resource observations. Baseline, original
capture and guard records are reused from the candidate side of
[matched_reverse_transport_forward_20261019](../matched_reverse_transport_forward_20261019/README.md).
The new manifest pins that dependency's complete manifest and archive hashes;
the reader rechecks all dependency members before using them. No baseline is
reconstructed from current mutable source. Large ELFs, checkpoint payloads and
toolchain dependencies remain external; this is not a hermetic rebuild.

Reproduce with the retained original `baseline/job/spec.json` (in the dependency)
and `candidate/job/spec.json` command arrays, fresh task-local build directories,
the original selected source revisions and pinned inputs/tools. Both retain
prepacking, shared activation preparation, connected VPU/broadcast operations,
word-group preparation, guarded host math, movement-bounded tiles and reverse
BF16 transport. Add `--reuse-batch-rhs` only to the candidate. After actual
termination, run `tools.validate_completed_merlin_stage` on each result and
`tools.compare_matched_tinyllama_forward` with the original independent source
and token-lookup boundary archives. A rebuild with newer sources is a new
experiment, not reproduction of these exact ELF hashes.
