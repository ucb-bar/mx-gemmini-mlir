# Matched full-forward shared activation preparation

Shared activation packing reduces full 160-token, 22-layer TinyLlama execution
by **9.5878%**, with every one of the 5,120,000 output words bit-identical to
the earlier numerically corrected compiler run. Both pass the original
Nicolás-relative numerical gate. The remaining source gap is **13.2398%**.

| Same token-to-logits boundary | Modeled cycles | Mean KL, late33 | Worst KL | Top agreement, late33 |
| --- | ---: | ---: | ---: | ---: |
| Independent Nicolás source | 12,762,360,525 | 0.0641713 | 0.1896031 | 27/33 |
| Earlier corrected compiler | 15,984,649,026 | 0.0637956 | 0.1699798 | 27/33 |
| Shared activation preparation | 14,452,067,124 | 0.0637956 | 0.1699798 | 27/33 |

This measurement adds explicit shared activation preparations to the earlier
numerical placement and connected-multiply candidate. Merlin proves SSA input
equivalence, read-only consumers, effect barriers and lifetime budgets. MX
supplies the packed representation, source-bound consumers and executable ABI.
The pass finds 44 groups and removes 66 duplicate preparations under a 1MiB
logical live-buffer budget. No operation is chosen by model name or provenance;
IDs bind the selected current source to its catalog.

All complete compiler logits are retained. The source-relative gate covers
the 33 complete independent-source rows that were retained, with the unchanged
5% mean/worst KL margin and top-token criterion. All 160 compiler FP32 top-token
agreements remain 133. The independent source retained 160 top tokens but only 33
complete rows. No full160-row source numerical claim is made.

Accelerator counters are identical: 3,775,124 commands, 490,737 fences,
29,340 VPU commands, 4,259,846,464 load bytes and 40,970,240 mesh tiles. This
optimization removes repeated host preparation; these extension counters do
not track host memory with `mem.host_tracking=0`. Static LLVM allocation-site
estimates decrease from 1,572,969,550 to 1,539,607,082 bytes; those sums are not
measured peak memory. Both runs select a 2GiB arena and 155 prepacked checkpoint
kernels. The linked ELF increases from 1,330,443,728 to 1,330,570,704 bytes.

The cycle ratio uses the retained source control with embedding lookup inside
its timer. The older precomputed-embedding timer has a different boundary.
The sigmoid-fusion, matrix/BF16-handoff and cached-session experiments have
separate runs; their completion and speedups are not inferred from this result.

The archive retains the complete new console, original successful stage and
full-output validation receipts, catalog, generated C/IR, shared-preparation
proofs, actual live environment, recipes and admission records. It reuses the
immutable [earlier corrected-run archive](../matched_forward_20261018/README.md)
for the baseline payload instead of duplicating it. Gigabyte-scale ELF and
checkpoint files are omitted; their observed identities remain recorded.
`verify.py` uses NumPy and the pinned earlier full-word reader, reopens both
complete outputs and the equal source timer, and recomputes the quality metrics.
With `--build-root TASK_TMP/mx_model_goal_20261018`, it additionally reopens both
actual compiler ELFs. Image admission is structural custody, not an instruction
policy or independent hardware qualification.

Reproduce from the configured task worktree by rebuilding the earlier corrected
candidate with `--share-activation-packing` and the identical capture, execution
policy, profile and extension. Keep prepacking, entry projection and connected
multiplies selected; use a full `--output-dump-cap 5120000`, explicit final ELF
admission and the original arena/timing settings. Run
`tools.validate_completed_merlin_stage`, then `tools.compare_matched_tinyllama_forward`
with the earlier corrected build as baseline and
`--boundary-archive docs/evidence/nicolas_tinyllama_lookup_equal_boundary_20261015`.
The archive retains the exact execution policy and compiler source snapshots.
