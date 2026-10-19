# Quality-qualified connected TinyLlama forward

The completed, uninstrumented 160-token, 22-layer forward takes
**13,735,943,708 modeled cycles**. It uses **4.9552% fewer cycles** than the earlier
shared-preparation result and **14.0679% fewer cycles** than the numerically corrected
compiler baseline. All **5,120,000 output words are bit-identical** to both.
The remaining gap to the independent reference implementation is **7.6286%**.

| Same token-to-logits boundary | Modeled cycles | Mean KL, late33 | Worst KL | Top agreement, late33 |
| --- | ---: | ---: | ---: | ---: |
| Independent reference source | 12,762,360,525 | 0.0641713 | 0.1896031 | 27/33 |
| Numerically corrected compiler | 15,984,649,026 | 0.0637956 | 0.1699798 | 27/33 |
| Shared activation preparation | 14,452,067,124 | 0.0637956 | 0.1699798 | 27/33 |
| Connected sigmoid chains and BF16 handoffs | 13,735,943,708 | 0.0637956 | 0.1699798 | 27/33 |

The existing source-relative gate retains its 5% mean/worst KL margin and
top-token criterion. Independent full source logits were retained only for the
33 late rows; no full160-row source distribution claim is made. All compiler
logits are retained, and FP32 top-token agreement remains 133/160.

This candidate adds 22 connected sigmoid/multiply chains and 44 private BF16
matrix-to-VPU handoffs to the earlier shared-preparation run. The 4.9552% result
is their combined full-forward effect, not an isolated sigmoid speedup.
Connected pointwise intermediates remain resident within each admitted chunk,
with every required BF16 rounding step preserved. Matrix handoffs retain BF16
in ordinary memory; they do not retain mesh accumulators in scratchpad.
Runtime guard and fallback costs are included. Fast-path coverage was not
separately instrumented, and the sigmoid selection remains experimental.

Commands fall from 3,775,124 to 3,542,604 and load bytes from 4,259,846,464 to
4,227,602,752. Fences increase from 490,737 to 495,601; VPU commands increase
from 29,340 to 29,364. Mesh tiles remain 40,970,240. No reduction in fences or
arithmetic work is inferred from the elapsed-cycle improvement.

The static LLVM allocation-site sum falls from 1,539,607,082 to 1,431,470,834
bytes, with 22 fewer allocation sites. This is not measured peak memory; both
builds retain the 2GiB arena. The ELF grows from 1,330,570,704 to 1,330,824,360
bytes. Host memory tracking is off under the same pinned simulator profile.

The archive retains the complete new native console, original successful stage
and full-output validation receipts, generated C/IR, catalog, source bindings,
chain/transport proofs, actual live environment, recipe and final ELF admission.
It reuses the immutable
[shared-preparation archive](../matched_shared_activation_20261018/README.md)
for the baseline. Large checkpoint/ELF files remain in the task workspace and
their observed identities are retained. The actual compiled quantizer C and LUT
are retained with hashes matching the catalog; full transitive tool/header
closure is not claimed. Structural admission is not independent hardware or
forbidden-instruction-policy qualification.

The actual endpoint predates the explicit `gate_first` spelling. A retained
strict source-matcher/emitter check rematches all 22 chains and regenerates the
actual C byte-for-byte using their actual endpoint names. This is a naming
difference, not a substituted arithmetic implementation.

Run `python3 docs/evidence/matched_connected_forward_20261019/verify.py` with
NumPy available. It reopens every archive member, all complete compiler words,
the equal source timer and unchanged numerical gate using the pinned earlier
reader. Add `--build-root TASK_TMP/mx_model_goal_20261018` to reopen both actual
compiler ELFs too. Earlier evidence remains unchanged.

Reproduce the shared-preparation full forward with identical capture, policy,
checkpoint data and simulator settings; add
`--connect-vpu-sigmoid-multiplies --matrix-vpu-transport`, retain connected
multiplies and checkpoint prepacking, and dump all 5,120,000 output words.
Do not select `--op-profile`. Run `tools.validate_completed_merlin_stage` with
the standard `mx_stage_output_validation.json` output name, then
`tools.compare_matched_tinyllama_forward` against the shared-preparation build,
using the retained independent source archive and
`--boundary-archive docs/evidence/nicolas_tinyllama_lookup_equal_boundary_20261015`.
The virtual-padding decoder and cached/SmolVLA sessions are separate experiments.
