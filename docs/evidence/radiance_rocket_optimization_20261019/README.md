# TinyLlama forward optimization measurements

These runs execute the complete 160-token TinyLlama forward: embedding lookup,
all 22 decoder layers, the LM head and all 5,120,000 F32 logits. The same
checkpoint, token IDs, FP8 matrix formats and pinned Rocket MX+VPU timing
configuration are used by the [Radiance-derived port control](../radiance_rocket_forward_20261019/README.md).
The source is `radiance-kernels` at
`5866983682583486b38ae64a4d1bfa5e7b2a002e`, adapted to Rocket as described in
the [port guide](../../radiance_rocket_model_baseline.md).

The fastest completed configuration takes **6,099,702,127 modeled cycles**,
versus **7,082,331,012** for the port. Its raw cycle ratio is **1.161×**,
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
| Specialized packing and streamed BF16 RMS | 9,057,532,733 | 0.0821650 | 133/160 | Fails full mean KL |
| Hybrid softmax, host score scaling and streamed BF16 RMS | 7,218,220,770 | 0.0818391 | 132/160 | Fails full and late mean KL |

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

Streamed BF16 RMS preserves its selected BF16 sequence. It saves
**825,060,684 cycles** in the original placement and **806,923,938 cycles**
in the hybrid-softmax/host-score placement. Full logits remain identical to
the corresponding unfused configurations. These are measured alternatives,
not percentage savings to add together.

All compiler runs execute **40,970,240 matrix tiles** and **655,523,840 matrix
busy cycles**. The port executes 41,308,160 tiles, 0.825% more, because its
attention key axis is padded to 256 rather than 160. That also makes its
probability/EXP packing axis 60% wider. Whole-model ratios include this
implementation difference; they do not isolate compiler fusion alone or
establish the fastest possible handwritten Rocket program.

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

The [placement policies and commands](../../../examples/models/optimization_policies/README.md)
and [target configuration guide](../../configure_target_and_precision.md)
provide reproduction inputs and flag contracts. The historical fixed release
registry remains unchanged. These are modeled Spike cycles with host memory
tracking disabled, not FireSim measurements or a dataset quality evaluation.
