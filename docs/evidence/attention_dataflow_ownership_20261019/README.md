# Generic attention relationships belong to Merlin

The QK→softmax→PV source relationship analysis now lives in
`merlin.frontends.attention_dataflow`. MX retains a lazy compatibility entry
point that delegates to it. Target-specific precision proposals, contraction
legality, physical layouts, numerical policy and executable implementations
remain in MX. Standalone imports do not eagerly load the frontend.

This corrects ownership of an existing target-independent graph analysis. It
does not introduce a new fusion or performance claim. The analysis relates
semantic source tags through unique SSA producer paths, using labels only as
identities. It does not establish the arithmetic bodies, alias/lifetime safety,
numerical equivalence, device capability or legality of moving operations.
Those remain separate requirements for executable transformations.

The complete non-docstring Python AST is identical to the frozen implementation
from MX revision `380fc56`. On seven exact retained inputs, the frozen old
implementation, new core implementation and actual MX delegate all return
identical ordered relationships without mutating the inventory:

| Input | Recognized pairs |
|---|---:|
| Archived TinyLlama capture | 2 |
| Archived DeepSeek capture | 1 |
| Archived Gemma capture | 2 |
| Archived SmolVLA capture | 14 |
| Current full 22-layer TinyLlama capture | 22 |
| Current SmolVLA prefix generic read view | 17 |
| Current SmolVLA flow generic read view | 16 |

These are current recognizer results, not a claim that every source attention
region is recognized or device-legal. `report.json` pins all input sources, both
implementations, the actual delegate, and the executable AST comparison.
The input identity list is reused from the sibling
[semantic placement check](../semantic_policy_placement_20261019/README.md).
The frozen old source is archival and never imported by production code.

Thirteen core tests cover unrelated/misleading labels, absent or arbitrary module
names, unmodified inputs, missing/wrong semantic tags, shape mismatch, operand
orientation, disconnected equal-shape operations, ambiguous contractions,
ambiguous softmax regions, decomposed same-region softmax, and single-pair reuse.
Thirty-nine target integration tests cover the delegate, real model inventories,
precision planning and complete default placement. Core documentation, structure,
target-name and staged-file regex gates and focused Ruff checks pass. The full
regex scan reports eleven existing calls in four unchanged files; their exact
HEAD/worktree byte identity is checked, and no allowlist or ratchet is changed.
No source capture,
saved policy, runtime image or simulation was modified.

Recompute all recorded comparisons from the configured task environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
  python docs/evidence/attention_dataflow_ownership_20261019/replay.py
```

This requires the recorded Merlin/MX source and Python with xDSL. The four
smaller captures are tracked in MX; the three larger original task source/read
views remain external at their recorded paths. Changed source or input bytes
refuse replay. The archive contains the small frozen old analysis and source
observations, without duplicating existing model captures or executables.

From the respective worktrees, the focused tests are:

```sh
# Merlin, with its src/ on PYTHONPATH:
python -m pytest merlin/tests/ir/test_attention_dataflow.py \
  --basetemp="$TMPDIR/pytest-attention-dataflow-reproduction" -q

# MX, with Merlin's src/ and MX on PYTHONPATH:
python -m pytest tests/test_merlin_attention_pairs.py tests/test_merlin_inventory.py \
  tests/test_model_precision_plan.py tests/test_execution_policy.py \
  --basetemp="$TMPDIR/pytest-attention-provider-reproduction" -q
```

Set `TMPDIR` and `PYTEST_DEBUG_TEMPROOT` to the task's `.tmp` and use fresh test
directories. Existing whole-model measurements retain their frozen binaries;
this source-only relocation does not supply another full-model result.
