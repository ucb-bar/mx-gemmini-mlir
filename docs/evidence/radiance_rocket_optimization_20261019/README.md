# TinyLlama forward optimization measurements

These runs execute the complete 160-token TinyLlama forward: embedding lookup,
all 22 decoder layers, the LM head and all 5,120,000 F32 logits. The same
checkpoint, token IDs, FP8 matrix formats and pinned Rocket MX+VPU timing
configuration are used by the [Radiance-derived port control](../radiance_rocket_forward_20261019/README.md).
The source is `radiance-kernels` at
`5866983682583486b38ae64a4d1bfa5e7b2a002e`, adapted to Rocket as described in
the [port guide](../../radiance_rocket_model_baseline.md).

The fastest completed configuration takes **5,849,962,853 modeled cycles**,
versus **7,082,331,012** for the port. Its raw cycle ratio is **1.211×**,
but it fails the top-token agreement requirements. **This is not a qualified
speedup.** Both the full-160 and late-33 gates remain unchanged.

| Completed configuration | Modeled cycles | Full-160 mean KL | Full-160 top tokens | Numerical gate |
| --- | ---: | ---: | ---: | --- |
| Radiance-derived Rocket port | 7,082,331,012 | 0.0774637 | 132/160 | Control |
| Fixed compiler release | 10,326,818,367 | 0.0821650 | 133/160 | Fails full mean KL |
| Specialized FP8 packing | 9,882,593,417 | 0.0821650 | 133/160 | Fails full mean KL |
| Hybrid softmax | 8,558,679,644 | 0.0818391 | 132/160 | Fails full and late mean KL |
| Hybrid softmax, host score scaling | 8,025,144,708 | 0.0818391 | 132/160 | Fails full and late mean KL |
| EXP-only VPU placement | 6,099,702,127 | 0.0804881 | 129/160 | Fails full and late top tokens |
| EXP-only placement and grouped QKV completion | 6,099,669,667 | 0.0804881 | 129/160 | Fails full and late top tokens |
| EXP-only placement, grouping and streamed softmax | 6,115,876,839 | 0.0804881 | 129/160 | Fails full and late top tokens |
| Streamed softmax and FP32 sigmoid reciprocal | 6,227,350,452 | 0.0784700 | 130/160 | Fails full and late top tokens |
| FP32 sigmoid reciprocal and fused gate/up packing | 5,849,962,853 | 0.0784700 | 130/160 | Fails full and late top tokens |
| EXP-only placement and BF16 residual state | 6,283,478,981 | 0.0802911 | 132/160 | Fails late top tokens |
| BF16 residual state and FP32 sigmoid reciprocal | 6,394,946,919 | 0.0833883 | 130/160 | Fails full mean KL/top tokens and late worst KL |
| Original pass order: requested gate/up fusion does not apply | 6,394,934,285 | 0.0833883 | 130/160 | Same numerical failures as its unfused control |
| BF16 residual state and realized gate/up fusion | 6,017,860,547 | 0.0833883 | 130/160 | Same numerical failures as its unfused control |
| BF16 residual state, gate/up fusion and softmax launch guard | 6,001,616,063 | 0.0833883 | 130/160 | Fails full mean KL, top tokens; late worst KL |
| Launch guard and sequential FP32 RMS packing | 5,970,974,009 | 0.0833883 | 130/160 | Fails full mean KL, top tokens; late worst KL |
| Sixteen-lane FP32 RMS packing | 5,968,133,789 | 0.0803546 | 130/160 | Fails full top tokens; late worst KL, top tokens |
| Sixteen-lane RMS and softmax reductions | 5,995,071,328 | 0.0815260 | 135/160 | Fails full mean KL; late worst KL |
| Sixteen-lane reductions and direct SiLU division | 5,975,235,678 | 0.0864567 | 131/160 | Fails full mean KL, top tokens; late mean KL, worst KL |
| Specialized packing and streamed BF16 RMS | 9,057,532,733 | 0.0821650 | 133/160 | Fails full mean KL |
| Hybrid softmax, host score scaling and streamed BF16 RMS | 7,218,220,770 | 0.0818391 | 132/160 | Fails full and late mean KL |

Checkpoint prepacking and output dumping are outside both timed intervals.

The two RMS rows are alternative configurations. They are not additional
transformations applied to the EXP-only row. The archive preserves failed
quality results alongside faster timings.

## What accounts for the changes

Specializing FP8 packing saves **444,224,950 cycles**, or **4.30%**, while
preserving every original compiler output word. Switching to hybrid softmax
then saves **1,323,913,773 cycles**, with different intermediate arithmetic:
BF16 VPU EXP, FP32 max/subtraction/ordered sum/division. Moving the binary
attention-score scale to host code saves another **533,534,936 cycles** with
identical hybrid-softmax logits.

EXP-only placement retains VPU softmax and sigmoid, while moving normalization,
reductions, RoPE arithmetic, residual adds and other elementwise operations
to FP32 host code. That configuration saves **1,925,442,581 cycles** relative
to the preceding host-score row. This combines placement and precision
changes; it is not a bit-preserving fusion. Its two KL limits pass on both
position sets, but top-token matches fall to **129/160 and 26/33**, against
**132/160 and 27/33** for the port.

Grouping the 22 QKV projection triplets saves only **32,460 cycles** in the
complete EXP-only execution, about **0.00053%**, with identical full logits.
It removes 44 completion fences; matrix commands, matrix work and transfer
bytes are unchanged. The standalone projection probe's small improvement
does not translate into a material model-wide gain.

Streamed softmax regresses by **16,207,172 cycles** against the grouped
configuration, despite a small standalone-probe improvement. Splitting work
at each attention matrix changes EXP launch packing from 101 to 128 launches
per layer, adding **594** launches across 22 layers. Complete logits remain
identical. MX revision `8df46f8a` adds a shape-based cost guard that rejects
this fusion when it increases launches. The complete guarded execution
takes **6,001,616,063 cycles**, saving **16,244,484** against the BF16-residual
gate/up-fused control while preserving every logit word.

Using FP32 reciprocal arithmetic for sigmoid improves full mean KL to
**0.07846997** and full top agreement to **130/160**. Late agreement remains
**26/33**. Both KL limits pass on both position sets, while both top-token
gates fail. This changes arithmetic; it is not an accuracy-preserving fusion.

Fusing the 22 gate/up pairs, hybrid sigmoid, multiplication and FP8 preparation
saves **377,387,599 cycles**, or **6.06%**, against that FP32-sigmoid run.
Every complete logit word is identical. This validates the fusion's numerical
preservation, while retaining the input configuration's failed quality gates.

Matching the port's BF16 residual-state format raises full agreement from
129/160 to **132/160** with BF16 sigmoid arithmetic. All full-160 limits pass,
but late agreement remains **26/33**, one below the required 27/33. With FP32
sigmoid reciprocal, late agreement rises to **29/33**, while full mean KL,
full top tokens and late worst KL fail. Residual precision and activation
arithmetic therefore remain separate measured numerical choices.

When gate/up fusion is composed with matrix-to-BF16-residual transport, it
must run before transport rewrites the down-projection boundary. The original
pass order realizes zero gate/up fusions and differs from its control by only
12,634 cycles. The corrected order realizes all 22, saves **377,086,372 cycles**
against the unfused control, and preserves every complete logit word. The
archive and lowered-call check retain this unapplied control separately from
the actual fusion result.

The sequential FP32 RMS-to-FP8 fusion realizes 45 sites, including the LM
head. It takes **5,970,974,009 cycles**, saving **30,642,054 cycles (0.51%)**
against the guarded execution, with every logit word unchanged. The separate
[normalization probe](../f32_rms_packing_20261019/README.md) saves about 28%
against a materialized C control using the production packer. That local
control has different optimization opportunities from the original LLVM
model lowering; its percentage is not a whole-model projection.

The sixteen-lane RMS option, sixteen-lane softmax option and direct SiLU
`gate / (1 + exp(-gate))` option explicitly change FP32 rounding order. All
three complete executions remain faster than the port and fail some unchanged
numerical limits, as shown above. Matching a source kernel's reduction order
alone does not establish model-wide numerical equivalence.

Streamed BF16 RMS preserves its selected BF16 sequence. It saves
**825,060,684 cycles** in the original placement and **806,923,938 cycles**
in the hybrid-softmax/host-score placement. Full logits remain identical to
the corresponding unfused configurations. These are measured alternatives,
not percentage savings to add together.

The layout audit also reopens both actual executables and compares every
learned matrix after undoing the source's tiled/combined layouts. All
**155 FP8 code and E8M0 scale payloads are identical** in canonical coordinates.
The retained [weight-equivalence report](source-compiler-weight-equivalence.json)
records each checkpoint parameter and both tile partitions. Learned source
matrices use K steps of 256; these compiler matrices use 512. Tile scheduling
therefore contributes to the comparison as well as operator placement.

Hybrid softmax batches 51 rows per VPU launch. In the EXP-only configuration,
its 22 softmax sites use **2,222 EXP launches**, versus **112,640** in the
port's row-at-a-time schedule. The compiler's default BF16 sigmoid contributes
9,680 additional vector operations, for **11,902 VPU operations** in the
complete EXP-only run; the port uses 116,160 EXP operations overall. These
counts explain launch amortization without equating different phase timings.

All compiler runs execute **40,970,240 matrix tiles** and **655,523,840 matrix
busy cycles**. The port executes 41,308,160 tiles, 0.825% more, because its
attention key axis is padded to 256 rather than 160. That also makes its
probability/EXP packing axis 60% wider. Whole-model ratios include this
implementation difference; they do not isolate compiler fusion alone or
establish the fastest possible handwritten Rocket program.

The source's rotary coefficient tables are generated using FP64 calculations
before execution. The compiler computes angles from captured F32 frequencies
and calls target `sinf`/`cosf` during execution. These preserve the same model
parameters but differ in rounding and preparation costs. Together with
reduction order, activation arithmetic and residual precision, this is a
reason to apply the unchanged complete-output quality gates rather than
assume numerical identity between implementations.

## Verify the artifacts

```sh
python docs/evidence/radiance_rocket_optimization_20261019/verify.py \
  --out "$TASK_DIR/.tmp/forward-optimization-check.json"
```

Choose a fresh output path. The reader needs NumPy. It reopens the complete
source/control evidence and diagnostic profiles, then checks each candidate
archive and every observed logit word. It recomputes mean/worst KL and top
agreement on both position sets, applies the unchanged 5% KL/no-top-decrease
gates, and derives timing ratios from the actual console intervals. It also
checks input IDs, build identity, policy, target/environment, catalog-derived
matrix work, completion and receipt/admission/payload-audit bindings.

Each archive retains the actual console, comparison, execution receipt,
compilation/link recipe, generated C/LLVM/MLIR/catalog text, original command
and producer, structural admission and linked-payload audit. The primary
comparison reopened the actual ELF and direct compilation inputs. Separate
payload audits compared actual linked tokens/constants, all 155 fixed matrix
code/scale payloads, and retained embedding/normalization weights against the
capture and catalog.

Large candidate ELFs, objects, checkpoint payloads and toolchain binaries are
external inputs. The portable reader verifies their retained identities and
audit bindings; it does not reopen absent binaries or establish a hermetic
rebuild. The separate EXP-only rebuild record reproduces its original ELF
exactly from clean MX revision `d4797f2d`; it records build identity and adds
no execution measurement.

To repeat the learned-weight check with the actual large executables:

```sh
python -m tools.audit_radiance_compiler_weights \
  --port-build /path/to/radiance-forward-build \
  --compiler-build "$ARTIFACT_ROOT/tinyllama-forward-exp-build" \
  --out "$ARTIFACT_ROOT/source-compiler-weight-equivalence.json"
```

This check covers encoded learned matrices. Runtime activations retain the
selected placement and arithmetic differences described above.

The [placement policies and commands](../../../examples/models/optimization_policies/README.md)
and [target configuration guide](../../configure_target_and_precision.md)
provide reproduction inputs and flag contracts. The historical fixed release
registry remains unchanged. These are modeled Spike cycles with host memory
tracking disabled, not FireSim measurements or a dataset quality evaluation.
