# Projected stage entries in a recurrent session

This small real MX/Spike execution qualifies the session ABI plumbing needed
to reuse whole-model stages after removing dead immutable checkpoint arguments.
It is not a TinyLlama or SmolVLA performance result.

The captured fixture has an identity linear projection with bias, a mutable
state input, and two observable results: logits and next state. Prefill runs
once, followed by three recurrent decodes with different streamed inputs.
Values vary across rows and columns and are exactly representable under the
selected FP8/BF16 arithmetic. Decode's captured initial state is deliberately
`-7`; the actual prefill output must replace it before the first decode.

All **4,096 logit and state words** in both executions match an independent
NumPy reference bit for bit. The verifier reconstructs that reference from
the actual captured weights, inputs and stream frames. It does not accept
checksums as a substitute for comparing every dumped value.

| Fixture execution | Shared checkpoint bytes | Original step intervals | Reset/handoff intervals | Accounted cycles |
| --- | ---: | ---: | ---: | ---: |
| Original entry ABI | 4,224 | 391,917 | 2,816 | 394,733 |
| Projected entry ABI | 128 | 391,707 | 2,819 | 394,526 |

These are one-session modeled-cycle observations. The small timing difference
is not a general profitability result. Both variants prepack the same immutable
matrix; the storage difference removes its unused original FP32 copy. The
packed matrix remains in the linked device catalog. The allocator reservation
and logical network/state computation are unchanged.

Merlin owns the generic mechanism: rederive saved projections from typed SSA,
recheck every source/table pin, reprove namespaced entries and direct callers,
generate compact C tables, and map public captured input indices to retained
descriptors. Original state input 3 becomes descriptor 2; stream input 2
becomes descriptor 1. The session's logical bindings and output ordinals keep
their original meanings. Shared weight materialization reopens every selected
stage proof and still refuses different nonempty layouts.

MX owns the actual object correspondence, symbol renaming and final link.
Projected model objects must match their completed compilation recipe and
projection receipt. Every generated projected caller is rebuilt. Active MX
session links now require an explicitly selected final-ELF admission service;
these runs use the same existing **structural digest/declared-ISA service** as
the stage builds. This establishes no instruction-policy verdict. The runner
reopens the accepted report, image identity and selected policy source pins.

The original step timing intervals remain reported. Optional
`--measure-preparation` additionally measures reset and each cross-program
handoff; the reader refuses missing, reordered, duplicate or inconsistent
records. `execution_total_performance` sums these phases and the original steps.
Console output and phase bookkeeping are excluded explicitly. This is an
accounting of those compute segments, not host wall-clock latency. Old cached
step totals must not be described as including reset or cross-program copies.

The extension is the archived `5058ebfd…cdc5` image with
`GEMMINI_MODE=both`, `e4m3_vpu`, and `mem.host_tracking=0`. Captures, complete
logs, generated callers/adapters, compact blobs, actual ELF/model-object bytes
(compressed), accepted admission reports and receipts are retained here.
Portable verification needs only Python and NumPy:

```bash
python docs/evidence/projected_session_20261018/verify.py
```

The portable check compares outputs and reopens the archived image/object
digests. Recomputing typed projection legality uses the Merlin tests or a
fresh reproduction; the archive does not reconstruct omitted transitive
toolchain/provider dependencies from their names.

Run [reproduce.sh](reproduce.sh) from the MX repository with a fresh task-local
output directory. Set `TMPDIR` to the task's `.tmp`, `CAPTURE_PYTHON` to the
capture interpreter, `MERLIN_COMPILER_PYTHON` to the compiler interpreter with
torch-mlir, and `LLVM_INSTALL`/`RISCV_ROOT` to the selected toolchains. No old
artifact is overwritten. The script runs both actual linked sessions, with
complete bounded output dumps and preparation accounting enabled.

Full 160-token TinyLlama baseline/candidate runs were still active when this
fixture was qualified. The 128-token prefill plus 32 cached decodes and the
remaining global optimization areas remain part of the active goal.
