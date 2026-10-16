# Full-model MX+VPU comparisons on Spike

The full-session matrix-issuer substitutions for TinyLlama and SmolVLA hold
Merlin's host graph and VPU objects fixed. They measure the MX issuer inside
each model session. They do not compare an independent Nicolas application
with a compiled application.

| Captured session | Precision | Compiler cycles | Nicolas issuer cycles | Compiler difference |
| --- | --- | ---: | ---: | ---: |
| [TinyLlama, 1 prefill + 32 decodes](evidence/nicolas_matrix_substitution_tinyllama_full_20261014/README.md) | Mixed FP4/FP8 | 83,547,564,353 | 83,544,602,795 | +0.004% |
| [SmolVLA, 1 prefix + 10 flows + 1 action](evidence/nicolas_matrix_substitution_smolvla_all_fp8_20261014/README.md) | All FP8 | 180,219,637,915 | 179,451,403,096 | +0.428% |

These are fair **issuer-only** comparisons: both programs execute the same
captured model graph with equal output digests. The percentages measure only
the effect of replacing compiler MX command issuance with Nicolas's native
issuer. They do not measure the cost of the compiler's whole-model schedule
against an independently written schedule.
The TinyLlama pair matches all 99 recorded output digests; the SmolVLA pair
matches all 33. The all-FP8 SmolVLA final action has cosine 0.988524 against
the captured FP32 trajectory. That accuracy result applies to both sides of
its issuer pair because their outputs are identical.

## Independent TinyLlama forward

The existing one-pass control runs all 22 decoder layers and the LM head on
160 token IDs. It uses Nicolas's generated source program and the compiler's
model2MLIR capture with the same IDs, all 200 FP32 checkpoint tensors, and
all-FP8 matrix policy. The source program originally began with precomputed
BF16 embeddings, so its cycle count excluded the token lookup performed by
the compiler.

`tools/prepare_nicolas_tinyllama_lookup.py` adds the checkpoint's full FP32
embedding table and the 160 token IDs to the source program. It verifies that
the lookup and BF16 rounding exactly reproduce all 327,680 captured source
embedding values. The lookup executes after Nicolas's stage timer starts.
`tools/run_nicolas_tinyllama_lookup.py` builds the adapted source with the
pinned RV64 toolchain, runs the same `e4m3_vpu` Spike timing model, and binds
the generated ELF and log to their inputs. The source payload must match the
archived source-blob hash before the build runs.

`tools/compare_fair_tinyllama_onepass.py` checks the new source's 33 late
BF16 logit rows against the original independent source run and compares its
cycles with the archived compiler run. It reports both programs' late-token
and full-forward top-token agreement with the same FP32 capture. The cycle
ratio covers equal logical input-to-output work; it is **not** an equal-quality
speedup when those accuracy results differ. The source and compiler use different
host/VPU policies, which are part of their implementations.
The comparator also checks all 160 source top tokens against the generated
BF16 source reference when its payload is available.

The [equal-boundary one-pass archive](evidence/nicolas_tinyllama_lookup_equal_boundary_20261015/README.md)
passes those checks. Nicolas's independent source takes **12,762,360,525**
modeled cycles; the compiler takes **15,170,996,572**, a **+18.873% raw
compiler cycle difference**. Their top tokens agree at 114/160 positions.
The source matches the captured FP32 top token at 131/160 positions and the
compiler at 103/160. The source's full output top-token list matches its
generated BF16 reference, and all 33 late BF16 logit rows equal the earlier
source run. The two programs therefore have matched logical inputs and
outputs to measure, but **not equal numerical quality**. Neither the cycle
ratio nor the issuer-only controls establish a full-session, equal-quality
compiler-versus-handwritten speedup.
Using float64 softmax on the archived 33 late rows, Nicolas's source has
mean KL **0.0641713** and worst-row KL **0.1896034** nats against the captured
FP32 logits, with 27/33 top-token matches. The 5% numerical gate therefore
requires compiler mean KL at most **0.0673798**, worst-row KL at most
**0.1990836**, and at least 27/33 top-token matches for these same rows.
The current compiler one-pass log records a full-output digest and top tokens,
but not all logit values, so its KL gate is not yet measurable from that run.
On the separate archived 64-token prefill, the selected-VPU compiler control
has last-row KL **0.15506** against Nicolas's **0.07090**. Its
[quality receipt](evidence/tinyllama_full_safe_vpu_20261014/quality_gate.json)
fails the 5% gate despite matching the top token. That one-row result does
not measure the newer per-site policy or a cached session.
The older paired MX+VPU source and compiler prefill has source KL **0.07090**
and compiler KL **0.19283**. Its [receipt](evidence/nicolas_tinyllama_full_source_vpu_20261014/quality_gate.json)
also fails; differences in host algorithms and setup still prevent using
its cycle ratio as a matched performance result.

The source-generation input recipe and pinned revisions are in the
[one-pass evidence](evidence/nicolas_tinyllama_onepass160_same_input_20261014/README.md).
Keep the generated payload, object files, and ELF in a task scratch directory;
the repository stores compact receipts and logs.

## Cached TinyLlama and SmolVLA sessions

The captured TinyLlama session is a 128-token prefill followed by 32 cached
one-token decodes with mixed FP4/FP8 matrix bindings. Nicolas's existing
full-model source is a one-pass all-FP8 forward. An independent session
baseline needs the same cached KV state, one-token source schedule, and
precision policy before its total cycles can be compared with the compiled
session.

The captured SmolVLA session contains one prefix encode, ten flow steps, and
one action decode. Nicolas's source roster has attention and projection
kernels, but no complete SmolVLA application. A source session must stitch
those kernels with the remaining host work, masks, and recurrent state. The
existing head-level cutpoints and full-session issuer swap remain separate
controls until that source application completes.

For either full session, publish the same captured inputs, checkpoint, target
profile, timing preset, timed boundaries, output-quality metrics, and
per-stage cycle totals for both programs. An equal-quality performance claim
also requires an agreed numerical acceptance criterion.
