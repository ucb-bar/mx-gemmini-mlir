# Default placement follows arithmetic, not provenance-label spelling

The default policy previously recognized an elementwise add by searching for
`#add` in its region label. A source region named `opaque#add_metadata` therefore
sent a verified multiply, subtraction or division to the host. This violated
the rule that provenance IDs bind source identity without selecting strategy.

The policy now uses the selector's checked `operation == "add"` classification
from the same eligibility analysis. Public `eligible_sites` retains its original
three-result interface. Labels remain source-binding keys. Target capabilities,
numeric guards, explicit policy validation and executable endpoints are unchanged.

Twelve regression cases cross four actual scalar operations with three unrelated
or misleading labels. The original policy fails exactly three cases; the corrected
policy passes all twelve. The focused policy, singleton-preparation and elementwise
lowering suite passes **37 tests**. The logs retain the actual failing and passing
runs. Earlier fixture-construction errors are not treated as bug reproduction.

The actual old and corrected default constructors also produce identical full
policy dictionaries on seven retained inputs: four archived model captures,
the current 22-layer TinyLlama capture, and current SmolVLA prefix and flow generic
read views. `report.json` pins every input, both policy sources, target profile
and resulting canonical policy digest. This is a source/policy compatibility
check, not a rebuilt executable, timing measurement or new numerical-quality
claim. The frozen old source comes from MX revision `3662440`; it is retained
only for comparison, not imported by production code.

From the configured task environment, run:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
  python docs/evidence/semantic_policy_placement_20261019/replay.py

TMPDIR="$PWD/../.tmp" PYTEST_DEBUG_TEMPROOT="$PWD/../.tmp" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../merlin/src:../model2mlir:. \
  python -m pytest tests/test_execution_policy.py \
    tests/test_merlin_singleton_prepare.py \
    tests/test_merlin_vpu_elementwise_lowering.py \
    --basetemp="$PWD/../.tmp/pytest-semantic-policy-reproduction" -q
```

Use a fresh task-local test directory. Replay requires the original source input
paths recorded in the report, Python with xDSL/NumPy/pytest, and the recorded
Merlin/MX source environment. It executes the frozen old constructor and current
constructor, compares their complete results, and checks all retained hashes.
The four smaller capture inputs are already tracked in MX; large current task
captures/read views remain external. No original capture, saved policy, image or
running simulation was changed. Existing measured images remain frozen evidence.
