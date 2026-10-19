# Read immutable captures through a generic printer view

Some captured MLIR uses custom assembly syntax that xDSL cannot read directly.
The original capture and its receipts remain unchanged. Merlin can retain an
explicit native generic printer view outside the capture, with separate source,
view, interpreter and printer-script identities. It applies no rewrite passes.

For structural inspection, run `tools.audit_model2mlir_session` with a fresh
`--parse-workdir` outside the captured bundle. The audit first tries the original
text, then uses the existing native printer if needed. Each stage reports its
original `source_mlir_sha256`, actual `inventory_input_sha256` and optional
serialization-view record. The complete bundle inventory is rechecked after
the audit. This remains a structural inventory, not executable placement.

For a stage build, select `--generic-input-view`. The builder creates and
rechecks its own view under `BUILD/input_read_view/`, uses it for source
inventories and placement-policy validation, and compiles the original captured
model. The receipt retains both source identities and the exact printer receipt.
A selected execution policy must bind the generic view's SHA; source-operation
coverage and all final-image and numerical checks remain in place.

Create a policy with the same printer and original stage input:

```python
from pathlib import Path
import json
from merlin.llvmlower.generic_form import create_generic_read_view
from mx_gemmini_support.execution_policy import make_policy
from mx_gemmini_support.target_profile import load_profile

view = create_generic_read_view(Path(STAGE) / "model.mlir", FRESH_VIEW_DIRECTORY)
policy = make_policy(Path(view["view"]["path"]).read_text(), load_profile(Path(PROFILE)))
Path(FRESH_POLICY_FILE).write_text(json.dumps(policy, indent=2) + "\n")
```

Then pass that policy and `--generic-input-view` to `tools.build_merlin_stage`,
along with the ordinary compiler, profile, admission and execution options.
Changing source, view, printer, interpreter or selected receipt refuses the
build result. This transport supplies no numerical or performance qualification;
complete output and model-specific quality evaluation remain necessary.
