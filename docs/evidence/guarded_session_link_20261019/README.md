# Explicit guarded math in a complete recurrent session

The session linker now preserves an explicitly selected exact host-library
contract. Previously it linked model and device objects without the stage's
separate host-math wrapper. A guarded stage now requires the same explicit
`--host-math-guard guard.json` session selection. Incompatible contracts refuse.
Native sessions keep their existing link path.

GNU wrapping applies to all calls to the selected symbol in the final image,
including native stages. The selected contract must preserve returned bits,
errno and floating effects in its domain and retain the original fallback.
Merlin builds one common wrapper. The target session driver resolves and pins
the actual library and records the final link's defining suppliers for both
endpoints. The runner reopens the completed recipe, wrapper, library and
suppliers before launching Spike. The original admission checks still apply.

A captured two-program fixture executes one prefill and three recurrent decode
steps. Each step has a matrix, residual, host softmax and state feedback. Both
versions use identical compiled stage objects; only the session math selection
changes. The actual native and guarded executions produce eight identical
output/state hashes and four byte-identical 2,048-byte output tensors.

| Inclusive work | Native cycles | Guarded cycles |
|---|---:|---:|
| Reset | 1,330 | 1,330 |
| Prefill plus preparation | 170,467 | 160,739 |
| Three decodes plus preparation | 507,343 | 478,159 |
| Total | 679,140 | 640,228 |

The reduction is 5.7296% for this small fixture. Both runs use the same pinned
Spike, extension, profile and `both/e4m3_vpu/mem.host_tracking=0` settings. Timers
include reset, handoff and existing step intervals; console/bookkeeping remains
outside both. Commands 308, loops 4, fences 220, VPU commands 16, mesh tiles 16,
load bytes 38,912 and scale bytes 192 match. These are wider process counters.
Host tracking is off. This is neither a full-model speedup nor an independent
quality qualification. The broader endpoint/fallback checks remain in
[guarded_host_math_20261019](../guarded_host_math_20261019/README.md).

The actual guarded stages also refuse a link without the explicit session
selection. The final runner was exercised again with recipe/supplier reopening,
producing identical outputs and cycle counts. Thirty focused session/link/
comparison tests pass, including mismatched and missing contract refusals.

```sh
python docs/evidence/guarded_session_link_20261019/verify.py
```

The standalone stdlib verifier reopens all 492 archived files, complete logs,
all output dumps, ordered state records, inclusive timing, selected library,
actual final images, object identities and defining-supplier traces. The archive
also includes the capture, four stage builds, both linked images, selected
source snapshots, commands and negative-link log. Toolchain executables and
their dependency closure remain external. This is custody and completed native
execution evidence, without a hermetic rebuild or independent instruction proof.

With the environment/toolchains from the model guide, reproduce the fixture in
a fresh task-local directory:

```sh
python -m tools.prepare_projected_session_fixture \
  --out-dir "$TMPDIR/guarded-session-capture" --profile "$MX_PROFILE" --host-softmax
```

Build its prefill/decode stages using their generated policies, prepacked/pruned
weights and structural admission. Build the guarded pair with the reviewed guard
from the endpoint archive. Link each pair using `tools.link_merlin_session`,
`--measure-preparation`, both `--dump-output prefill:0` and
`--dump-output decode:0`, and structural admission. Add the explicit
`--host-math-guard` to the guarded link. Execute each complete session with
`tools.run_linked_merlin_session` and compare with
`tools.compare_linked_session_cycles`. Exact original command arrays are
retained as `fixture/*.command.json` in the archive; substitute fresh paths.
