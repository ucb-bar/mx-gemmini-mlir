# Independent session checks

The 160-token TinyLlama forward has a qualified independent Rocket MX+VPU
comparison: 7,082,331,012 cycles for the Radiance-derived port and
6,264,147,801 for the compiler, or **1.131×**. The published cached TinyLlama
and SmolVLA numbers compare compiler versions. They must not be presented as
speedups over independently stitched applications.

## Checks of the existing results

The October 20 audit reopened the forward evidence, recomputed its numerical
gates, and checked the actual winning executable's inputs, retained host
weights, 155 packed matrices and generated source identities. It also checked
the complete cached and SmolVLA output streams and both sides' final ELFs.

Two cached session ELFs and several intermediate files were missing from the
retained build directory. They were restored from archived code, identical
retained checkpoint data and the original link commands. Every final object
and executable had to match its original SHA-256. This restored the original
images; it did not create new timing observations. An initial recovery attempt
omitted a retained shim object and failed the catalog hash check. That failed
attempt remains reproducible from retained fragments and checkpoint data.

Separate checks derive recurrent state from model semantics:

- TinyLlama: all 22 layer positions advance from 128 through 160; all 33
  complete K/V state hashes differ; all 33 complete vocabulary outputs survive.
- SmolVLA: the prefix K/V cache remains identical through ten flow steps;
  each timestep matches repeated FP32 addition of `-0.1`; the ten action
  tensors differ; action decoding returns the final flow tensor exactly.

Hash changes alone do not prove that every K/V element is correct. These
checks supplement the numerical comparisons. The tests use a fixed input
stream and model-ready test observations; they do not establish corpus quality
or robot task success.

The TinyLlama capture labels its token source `synthetic_seed_0`. Its
`synthetic_tokens: false` field reflects reloading the token IDs from a file;
it does not establish natural-language corpus provenance. Treat this as a
synthetic input benchmark when reporting the results.

The [retained evidence](evidence/independent_session_checks_20261020/README.md)
includes a portable verifier. The original complete-session verifiers also
remain available for [cached TinyLlama](evidence/cached_compact_rhs_session_20261019/README.md)
and [SmolVLA](evidence/smolvla_latest_session_20261019/README.md).

## Independent cached execution

The cached port adds an independent descriptor scheduler and persistent K/V
state to the existing Radiance-derived Rocket kernels. It performs a prefill
and then one logical token per decode. Native matrix operations use sixteen
physical rows for a single-token decode. Cache reset, embedding lookup,
cache updates and output widening are timed. Console output is excluded.
Immutable weight packing and RoPE tables are prepared before execution.
The source scheduler specializes the fixed sequence positions and keeps one
persistent cache; the compiler measurement includes its functional stage
handoffs. These implementation differences are part of the complete-session
comparison.

V-cache quantization groups span 32 tokens. After an append, the port
requantizes the affected group from the original cache values. Reusing old
FP8 codes would lose information when the group's shared exponent changes.
The preceding quantized execution produces the cache used by the next step.
Floating reference caches are not injected into this trajectory.

Two smoke sessions, each with two layers, 64 prefill tokens and 33 single-token
decodes, pass a separate NumPy replay. They use two and eight query heads per
K/V head; the latter matches TinyLlama's head grouping. In each case all
17,408 output-logit words and 49,664 final K/V words match exactly. The replay
computes each matrix with the pinned Radiance mesh reference and each
exponential with the pinned Rocket VPU reference. It reconstructs the whole
packed cache independently after each append. Both tests cross a 32-token
cache-block boundary. They are not full TinyLlama performance measurements.

A second reference checks the eight-to-one case as a logical Llama graph,
using only the original prefill's parameter bindings. It does not execute the
decoder descriptors. Every logit and final K/V word matches. This checks the
new scheduler's attention-head assembly and cache updates independently of
its descriptor layout.

Use the repository and toolchain pins from the
[whole-model compilation guide](compile_e2e_models.md). The source kernel
revision is `5866983682583486b38ae64a4d1bfa5e7b2a002e`. The following command
expects the captured 128+32 TinyLlama session and the same pinned Rocket
profile, extension and startup files as the existing forward port.

`COMPILER_PREFILL_BUILD` and `COMPILER_DECODE_BUILD` are the original stage
build directories used to link the compiler session. `COMPILER_LINK` contains
its link receipt and executable; `COMPILER_RUN` contains its complete simulator
output and dumped logits.

```sh
python -m tools.qualify_radiance_rocket_cached \
  --radiance-root "$RADIANCE_ROOT" \
  --startup-dir "$STARTUP_DIR" \
  --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --extlib "$GEMMINI_EXTLIB" \
  --recorded-session "$CAPTURE" \
  --out-dir "$RUN/cached-port"

python -m tools.compare_radiance_rocket_cached \
  --port-dir "$RUN/cached-port" \
  --run-receipt "$COMPILER_RUN/mx_session_run.json" \
  --link-receipt "$COMPILER_LINK/mx_session_link.json" \
  --compiler-prefill-build "$COMPILER_PREFILL_BUILD" \
  --compiler-decode-build "$COMPILER_DECODE_BUILD" \
  --capture "$CAPTURE" \
  --out "$RUN/cached-comparison.json"
```

The complete matched run was launched after the smoke replay passed. Its
result remains pending. Before accepting a compiler speedup, the comparison
requires complete execution, matching model/input and hardware identities,
the expected accelerator work, actual measured ELF identities, and every
vocabulary output at all 33 positions. Mean and worst KL against the captured
floating reference must each be at most 1.05 times the source port's error;
top-token agreement must be at least as good. These criteria were fixed before
observing the full cached result. Row arithmetic differs between the source
port and the compiler, so bit identity between the implementations is not
required. Both use FP8/E8M0 groups of 32 with BF16 matrix outputs.

The comparison also undoes the source's weight tiling and checks all 155
learned matrices against the actual compiler executable for both prefill and
decode. Every FP8 code and E8M0 scale must agree; a shared checkpoint name
alone is insufficient. This preflight has already passed for the pending full
cached image. Full execution and its numerical gate remain pending.

To reproduce the small recurrent test, omit `--recorded-session` and use
`--decode-steps 33`. Add `--smoke-query-heads 16` to the build command for
the eight-to-one GQA case. Then run:

```sh
python -m tools.replay_radiance_rocket_cached \
  --build "$RUN/cached-smoke" \
  --radiance-root "$RADIANCE_ROOT" \
  --rocc-root "$PINNED_ROCC_ROOT" \
  --out-dir "$RUN/cached-smoke-replay"

python -m tools.replay_radiance_rocket_cached_graph \
  --build "$RUN/cached-smoke" \
  --radiance-root "$RADIANCE_ROOT" \
  --rocc-root "$PINNED_ROCC_ROOT" \
  --out-dir "$RUN/cached-smoke-graph"
```

## Remaining SmolVLA comparison

The pinned radiance-kernels revision supplies component kernels but no
complete SmolVLA session. An independent application still needs the vision
and language prefix, action expert, cache bindings, ten Euler updates and
action decoding. Substituting its matrix issuers into the compiler's host
program does not supply that baseline. The current **1.196×** SmolVLA result
remains a compiler-versus-compiler comparison, with all eleven dumped action
tensors identical. There is no independently stitched SmolVLA speedup to quote
yet.
