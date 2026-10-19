# SmolVLA repeated batch RHS applicability

The current repeated-RHS proof selects **zero of 56 prefix batch calls and zero
of 32 flow batch calls** in the completed optimized SmolVLA build. Enabling its
flag alone therefore does not establish another full-session optimization.
This is a structural analysis of the exact archived compiler IR, not a new
runtime measurement or proof that all further reuse is impossible.

| First refusal in the current proof | Prefix calls | Flow calls |
|---|---:|---:|
| Exact parallel-copy requirement | 40 | 8 |
| Matrix-axis-only transpose requirement | 16 | 24 |

The first category includes BF16-to-F32 scalar widening rather than only pure
copies. The second includes permutations between head and sequence axes, beyond
the proved matrix-axis swap. A full unary source trace retains scalar bodies,
types, affine maps and permutations for every call. Some traces reach repeated
broadcasts behind these operations; for example, a flow path expands
`1x163x5x1x64xbf16` to `1x163x5x3x64xbf16`, then reshapes, permutes and widens.
That observation alone does not prove contiguous groups at the final F32 native
input, nor authorize a compact ABI accepting BF16 input.

A future extension needs a general coordinate proof through those permutations,
an explicit compact conversion plan retaining widening/exceptional semantics,
and the existing privacy, ownership, completion and profitability checks. It
would then need native qualifiers and a new matched complete session. No such
extension or additional speedup is claimed here.

`report.json` records every inspected native call, its first proof refusal and
its source chain. `audit.py` reopens six original IR/catalog/binding members from
the sibling [complete SmolVLA archive](../smolvla_latest_session_20261019/README.md).
It verifies the archive and member hashes, catalog/object binding identities,
complete source-call coverage and source verification. It invokes Merlin's
actual current repetition proof, and compares the complete recomputed report
with the retained report, including hashes of the generic proof sources.

Replay from the original task, with Python containing xDSL and the recorded
Merlin source revision `6599207c8`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
  python docs/evidence/smolvla_rhs_applicability_20261019/audit.py
```

The report depends on the sibling archive; its large IR and catalog bodies are
not duplicated. A changed compiler proof or changed archived input causes replay
to refuse instead of silently retaining the old result. This evidence does not
alter numerical placement, source graphs, native ABIs or any running simulation.
