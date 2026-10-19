# Full TinyLlama operation profile

The completed 160-token, 22-layer forward attributes 94.53% of its instrumented
14,288,145,776 modeled cycles to operation intervals:

| Call/operation owner | Cycles | Share of total |
| --- | ---: | ---: |
| Matrix calls | 5,124,876,628 | 35.87% |
| VPU calls | 4,730,556,770 | 33.11% |
| Other host operations | 3,651,839,086 | 25.56% |
| Outside attributed intervals | 780,873,292 | 5.47% |

These are synchronous call intervals. Matrix and VPU intervals include their
host adapters, preparation and completion; they are not accelerator-only time.
The largest aggregate host family is normalization (1,965,299,172 cycles), led
by sensitive attention softmax operations retained on the host. Shared activation
preparation also remains material: its 2048-wide entry totals 1,028,561,756 cycles.

This build selects the earlier connected multiply pairs, 44 activation-sharing
groups and 44 private BF16 matrix-to-VPU handoffs. The later sigmoid fusion is
absent. Every one of its 5,120,000 output words is verified against the guest's
complete digest, yielding
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`,
the same digest as the earlier quality-qualified shared-preparation forward.

**Do not use this instrumented run for a clean speedup against reference or the
earlier uninstrumented compiler.** Marker calls and compilation perturb timing
and allocation. The verified uninstrumented result remains
[14,452,067,124 cycles](../matched_shared_activation_20261018/README.md).

Counters are 3,775,124 commands, 491,309 fences, 29,340 VPU commands,
4,259,846,464 load bytes and 40,970,240 mesh tiles. Host memory tracking is off.
The archive retains the complete native console, original successful build and
full-output validation receipts, actual live environment, operation table,
source/catalog bindings, preparation/handoff records and final ELF admission.
The large ELF remains in the task workspace; its identity is retained. This is
compiler instrumentation and structural image custody, not independent hardware
or forbidden-instruction-policy qualification.

Run `python3 docs/evidence/full_forward_profile_20261018/verify.py` to reopen all
archive members, rerun the pinned retained attribution reader and hash every
complete output word without extracting temporary files. Add
`--build-root TASK_TMP/mx_model_goal_20261018` to reopen the actual ELF too.

Reproduce the earlier shared-preparation full forward with the same original
capture, execution policy, immutable checkpoint data and pinned simulator;
add `--matrix-vpu-transport --op-profile` and retain the complete 5,120,000-word
output. Run `tools.validate_completed_merlin_stage` and
`tools.report_stage_profile`. Keep any corresponding performance comparison
separate and uninstrumented.
