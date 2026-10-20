# Configure the target and choose layer precision

Use the [model build guide](compile_e2e_models.md) to set up the pinned
repositories, Python environments, checkpoints, and toolchains. This guide
changes the hardware profile or the quantization policy for a new build.
The published model results use the exact policies in
[`examples/models/policies/`](../examples/models/policies/); changing a
policy produces a new numerical configuration.

## Select the hardware profile

The same `mx-gemmini-opt` binary handles the supported MX configurations.
Select hardware with a source-bound profile JSON and the build's `--profile`
argument. The profile describes the array dimensions, scratchpad and
accumulator limits, DMA resources, legal operand formats and PE modes, and
VPU capabilities. Precision choices must fit those capabilities.

The completed model workloads use
`MxE4M3Fp4VpuGemminiRocketConfig`: a 16×16 mesh, direct FP4 E2M1 and
FP8 E4M3 matrix operations, 32-element scale groups, and a BF16 VPU.
FP6 is available in other profiles, but this VPU profile has no FP6 compute
tuple. The [profile catalog](current_mx_profiles.md) describes the other
Rocket wrappers and standalone fragments, including their qualification
limits. A legal profile alone does not establish that every whole-model
adapter supports it.

Export a named configuration from the selected Gemmini checkout, including
its pinned MxGen submodule:

```sh
export GEMMINI_ROOT=/path/to/pinned/gemmini
export CONFIG=MxE4M3Fp4VpuGemminiRocketConfig
export PROFILE_DIR="$ARTIFACT_ROOT/selected-profiles"

"$CAPTURE_PY" -m tools.export_profiles --gemmini "$GEMMINI_ROOT" \
  --config "$CONFIG" --out "$PROFILE_DIR"
export PROFILE="$PROFILE_DIR/$CONFIG.json"
```

Use a fresh output directory. The exporter reads supported Scala
declarations; it does not elaborate arbitrary Scala. For the published
configuration, use the Gemmini and MxGen revisions in the
[release registry](../examples/models/release.json).

Check the complete profile against that checkout:

```sh
"$CAPTURE_PY" - <<'PY'
import os
from pathlib import Path
from mx_gemmini_support.target_profile import load_profile, profile_sha256

profile = load_profile(Path(os.environ["PROFILE"]),
                       rtl_root=Path(os.environ["GEMMINI_ROOT"]))
print(profile["name"], profile_sha256(profile))
print(profile["geometry"])
for mode in profile["legal_compute"]:
    print(mode)
PY
```

This check regenerates the profile and compares source revisions, source
hashes, and capabilities. Change the RTL configuration and re-export when
changing hardware. Editing capability fields in the JSON cannot enable a
missing hardware feature.

For a typed MX handoff, bind and verify it against the selected profile:

```sh
"$CAPTURE_PY" -m mx_gemmini_support.bind_profile \
  --mlir /path/to/handoff.mlir --profile "$PROFILE" \
  --rtl-root "$GEMMINI_ROOT" --out "$ARTIFACT_ROOT/profile-bound.mlir"
"$CAPTURE_PY" -m mx_gemmini_support.verify_profile_ir \
  --mlir "$ARTIFACT_ROOT/profile-bound.mlir" --profile "$PROFILE" \
  --rtl-root "$GEMMINI_ROOT" --mx-opt "$MX_ROOT/build/tools/mx-gemmini-opt"
```

The binder preserves the handoff's captured operand formats and
projections. For complete-model builds, pass `--profile` and `--rtl-root`
to `tools.build_merlin_stage`; the separate handoff command above is for
the typed dialect workflow. After changing profiles, regenerate the
execution policy and rebuild the stage objects and session executable.
The Spike extension or FireSim hardware must implement the selected profile.

## Find the compiler sites belonging to a layer

The executable backend uses a JSON `mx_gemmini.execution_policy.v1` or `v2`
policy. It binds every eligible matrix and VPU site to the exact captured
MLIR and profile hashes. Matrix decisions use `prov.region_id`, because a
layer can contain several contractions with different numerical roles.
The precision inventory also records the module's fully qualified name
(`fqn`), operation, role, and shape.

For example, use the TinyLlama cached-prefill inputs exported by the model
guide:

```sh
export INPUT_DIR="$ARTIFACT_ROOT/tinyllama-cached-inputs"
export STAGE="$INPUT_DIR/capture/stages/prefill"
export PRECISION_PLAN="$ARTIFACT_ROOT/prefill-precision-plan.json"
export POLICY_BASE="$INPUT_DIR/prefill-policy.json"
export POLICY_OUT="$ARTIFACT_ROOT/prefill-mixed-policy.json"

"$CAPTURE_PY" -m tools.plan_model_precision --reader merlin \
  --model-mlir "$STAGE/model.mlir" --profile "$PROFILE" \
  --rtl-root "$GEMMINI_ROOT" --out "$PRECISION_PLAN"

"$CAPTURE_PY" - <<'PY'
import json, os
from pathlib import Path

plan = json.loads(Path(os.environ["PRECISION_PLAN"]).read_text())
for site in plan["sites"]:
    print(site["region_id"], site["fqn"], site["role"],
          site.get("shape_mnk", site.get("shape_bmnk")),
          site["selected_format"])
PY
```

The plan proposes FP4 for eligible dense projections and FP8 for batched
contractions, including attention QK/PV. It is an eligibility inventory;
it does not change the build policy. The published execution policies
explicitly choose FP8 for every offloaded matrix and retain selected
sensitive operations on the host.

For a new capture without a published policy, generate a starting policy:

```sh
"$CAPTURE_PY" -m tools.make_execution_policy --stage "$STAGE" \
  --profile "$PROFILE" --preserve-singleton-reductions \
  --out "$ARTIFACT_ROOT/new-stage-policy.json"
```

This starts eligible matrices at FP8 and selects VPU placements from the
source operations. Review sensitive host/VPU boundaries before execution.
For an existing release capture, keep its exported policy as the starting
point so that its reviewed placement choices are preserved.

## Assign different precisions to different layers

Choose an exact region from the inventory. The example below changes one
eligible dense contraction to FP4 and preserves every other decision.
Repeat for the region IDs belonging to each layer you want to change.
Do not infer region IDs from layer names or reuse them across captures.

```sh
export FP4_REGION=replace-with-an-exact-region-id-from-the-inventory

"$CAPTURE_PY" - <<'PY'
import json, os
from pathlib import Path
from mx_gemmini_support.execution_policy import validate_policy
from mx_gemmini_support.target_profile import load_profile

plan = json.loads(Path(os.environ["PRECISION_PLAN"]).read_text())
policy = json.loads(Path(os.environ["POLICY_BASE"]).read_text())
region = os.environ["FP4_REGION"]
site = next(site for site in plan["sites"] if site["region_id"] == region)
if site["selected_format"] != "mxfp4":
    raise ValueError("choose a dense site that the inventory admits at FP4")
if policy["matrix_formats"][region] != "mxfp8":
    raise ValueError("the selected baseline site must currently use FP8")
policy["matrix_formats"][region] = "mxfp4"
source = (Path(os.environ["STAGE"]) / "model.mlir").read_text()
profile = load_profile(Path(os.environ["PROFILE"]),
                       rtl_root=Path(os.environ["GEMMINI_ROOT"]))
validate_policy(policy, source, profile)
with Path(os.environ["POLICY_OUT"]).open("x") as output:
    output.write(json.dumps(policy, indent=2, sort_keys=True) + "\n")
PY
```

The current full-model policy admits FP4 or FP8 at eligible dense sites,
and FP8 at supported batched sites. It does not admit FP6 or arbitrary
activation/weight format pairs. Both operands of a contraction use its
selected format. MX scales remain E8M0 per 32 elements; operands round
through BF16 and VPU operations remain BF16. Version 2 additionally permits
host placement for proven singleton reductions; `host` is not a general
matrix override. VPU decisions are separate `host`/`vpu` mappings under
`vpu_placements` for `softmax`, `add`, `elementwise`, `unary`, and `reduce`.

Unknown sites, incomplete policies, unsupported formats, and changed source
or profile hashes are rejected. Generate policies separately for prefill,
decode, prefix, and flow stages. Changing a quantization decision requires
repacking its constants and rebuilding affected objects; relinking an old
object with a new policy is insufficient.

## Build and check the changed policy

The optional arithmetic and fusion passes below are available together at
MX commit `a4c2cfbb`, with Merlin
`bb7c352a9e9c8ad085d24091fc83e4e7f5ff92ca` and model2MLIR
`f5e5c674a585bebbc2507dbd028ebf9d26e2afbc`. Keep the model guide's other
hardware and toolchain pins. The fixed release recipe predates these
options. Use a custom stage build and measure it separately; the
[retained kernel probes](evidence/mx_compiler_optimization_probes_20261019/README.md)
show their standalone effects and numerical checks.

### Select softmax arithmetic

Policy-selected VPU softmax has two implementations. The default `bf16`
rounds the input scores to BF16 and runs the complete softmax on the VPU.
`--vpu-softmax-implementation f32_reduce` instead computes the maximum,
subtraction, ordered sum and final division in FP32. It batches shifted BF16
inputs through the VPU exponential. This preserves FP32 normalization while
amortizing vector launches across several rows.

Set the relevant entries under `vpu_placements.softmax` to `vpu` in a copied
custom policy before selecting this option. Entries left at `host` retain
their original host softmax. The catalog records the chosen implementation
and rounding policy. Both choices need a full-output numerical comparison;
they are different arithmetic implementations.

### Select sigmoid arithmetic

`--vpu-sigmoid-implementation f32_reciprocal` retains BF16 VPU exponentials
and computes negation, addition and reciprocal in FP32. The default `bf16`
rounds every VPU step to BF16. This option applies only to policy-selected
FP32 sigmoid boundaries; surrounding multiplies retain their own placement
and arithmetic. The catalog records the implementation and rounding policy.

The FP32 alternative requires an explicit execution policy and cannot use
`--connect-vpu-sigmoid-multiplies`, whose resident implementation assumes
BF16 arithmetic throughout. It is an accuracy and placement choice. Measure
its cost and compare complete model outputs before selecting it.

### Stream private RMS normalization into packing

`--fuse-rms-packing` keeps a supported BF16 VPU RMS sequence in scratchpad and
packs each completed row tile into its FP8 consumer layout. It requires
`--vpu-matrix-transport` and that option's existing shared-packing, connected
multiply and integer-group prerequisites.

The pass checks the actual square, row sum, power-of-two division, epsilon
addition, reciprocal square root, gamma multiplication and packing calls.
It preserves their BF16 arithmetic boundaries. Static widths from 32 through
4096 that are powers of two are supported. Constants, readonly inputs,
destination ownership and complete intermediate privacy must match. A live
intermediate output or an unknown intervening effect keeps the original
sequence. Other precision formats retain their existing path.

The generated `mx_rms_packing_rewrite.json` lists each selected sequence and
its removed calls. The catalog retains the compiled endpoint and source
hashes. Check complete model outputs after compilation; a local packed-byte
test does not establish a complete-model performance improvement.

### Complete independent projections together

`--group-projections` groups two through eight immutable FP8 projections of
the same packed SSA input. Each projection retains its original weights,
matrix tiles and BF16 output. The generated endpoint writes disjoint BF16
destinations, completes the group once, then widens each output to FP32.
It does not concatenate checkpoint weights or reduce activation load traffic.

This option requires shared activation packing without retained unused RHS
arguments. The pass checks readonly input and private destination contracts,
matching tile partitions, an even native-loop count for each projection,
and the absence of intervening effects. The combined readouts must fit the
existing serial workspace reservation. Unsupported groups retain their
original calls. `mx_projection_groups_rewrite.json` records the groups that
were actually rewritten.

Measure this option separately: fewer completion boundaries can have little
effect when matrix transfers dominate. Grouped operation profiling is not
yet supported; use uninstrumented full-model timing and check every output.

### Consume softmax probabilities without a full FP32 intermediate

`--fuse-softmax-packing` combines a private hybrid softmax and its following
FP8 batch matrix call. It computes one matrix's probabilities, packs them
into a small private buffer, and consumes that buffer before continuing to
the next matrix. FP32 normalization, BF16 exponentials, FP8 block32 encoding,
matrix tiles and BF16 matrix readout retain their existing arithmetic.

This option requires `--vpu-softmax-implementation f32_reduce`, integer-group
packing and compact repeated RHS inputs. The pass checks exact batch shapes,
readonly inputs, private destinations and complete probability-use closure.
Only contiguous shape views and exact identity copies can disappear. An
observable probability tensor or an intervening effect retains the original
calls. Encoded RHS data is refreshed for every repeat group in every forward
call. `mx_softmax_packing_rewrite.json` records the selected pairs.

The option currently supports uninstrumented timing. Measure it on the target:
reducing intermediate storage does not by itself establish a speedup.

### Fuse gate/up projections through SiLU and packing

`--fuse-gated-projections` consumes the private BF16 gate/up matrix outputs
without materializing their FP32 readouts, sigmoid tensor or product tensors.
It retains both original matrix schedules and immutable weights, evaluates
`(gate * (1 / (1 + exp(-gate)))) * up` with BF16 VPU EXP and FP32 scalar
arithmetic, then writes canonical FP8 block32 codes and scales directly.

The option requires shared activation preparation, integer-group packing and
`--vpu-sigmoid-implementation f32_reciprocal`. The pass checks the actual
multiply order, matching packed input, tensor layouts, readonly contracts,
fresh destinations and complete intermediate privacy. It can replace either
an explicit preparation call or the packing inside a source-bound immutable
down-projection, using that matrix's existing prepared-input ABI. Unknown
effects, escaping tensors or unsupported shapes keep the original calls.

The combined matrix outputs and row buffers must fit the catalog's existing
serial workspace. Calls complete synchronously, with fresh state on every
invocation. `mx_gated_projections_rewrite.json` records each fusion, its source
kernels, sigmoid and terminal consumer. Operation profiling is not supported
for this option. The pass is available from MX commit `129cd185`; the Merlin
and model2MLIR versions above remain applicable. Use MX commit `a4c2cfbb` or
later when combining it with matrix-to-VPU transport: gate/up fusion runs
before the down-projection handoff. Compare all model outputs
against the unfused configuration before attributing a timing change to it.

For a stage build, pass the new policy explicitly:

```sh
"$CAPTURE_PY" -m tools.build_merlin_stage --stage "$STAGE" \
  --workdir "$ARTIFACT_ROOT/prefill-mixed-build" \
  --profile "$PROFILE" --rtl-root "$GEMMINI_ROOT" \
  --execution-policy "$POLICY_OUT" --merlin-src "$MERLIN_ROOT/src" \
  --llvm-install "$LLVM_INSTALL" --riscv-root "$RISCV_ROOT" \
  --prepack-checkpoint-weights --prune-unused-checkpoint-weights \
  --host-math-guard "$INPUT_DIR/guard.json" --arena-mb 2048 \
  --spike-arena-base 0xd0000000 --structural-elf-admission \
  --compile-timeout-s 6000
```

This is a custom stage build. To keep the published optimization options,
first generate the model guide's complete build plan without `--execute`.
Use its stage and session commands with fresh build directories and the new
policy paths. The `tools.model_release build` wrapper verifies the published
policy hashes and deliberately refuses a modified input export. Keep custom
policies outside that export and invoke the stage/link tools directly.

Inspect `mx_stage_probe.json` and `device_catalog/mx_catalog.json` for the
selected policy, actual matrix formats, placement census, and zero unbound
proposals. Link sessions through `tools.link_merlin_session` using the
rebuilt stages and the captured state bindings described in the model guide.
Check every state/output digest and complete selected tensor after execution.
The published output checker tests reproduction of the original policy;
a changed precision policy needs its own numerical comparison against the
same floating-point inputs and an independent quantized reference. Operand
fake quantization alone does not model the MX mesh arithmetic.

For a compiled 160-token TinyLlama forward, check the actual linked data too:

```sh
"$CAPTURE_PY" -m tools.audit_merlin_elf_payloads \
  --capture "$FORWARD_CAPTURE" --build "$FORWARD_BUILD" \
  --out "$ARTIFACT_ROOT/forward-payload-audit.json"
```

Set `FORWARD_CAPTURE` to its exported capture directory and `FORWARD_BUILD`
to the directory containing `model.elf`. The audit reopens compiled catalog
sources and checks every linked FP8 weight code and scale, the retained
FP32 embedding and normalization weights, and the embedded mask, rotary
constants and token IDs. It checks payload identity against the capture and
generated data files. Complete output and timing qualification remain
separate checks.

The [capture-time quantization handoff](quantization_handoff.md) documents
the separate TorchAO adapter, module-name overrides, FP6 codebook rules, and
typed handoff contracts. Those capture-time policies use a different schema
and do not replace the executable backend policy above.
