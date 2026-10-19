# Measured cached-state copy cost

The generated decode adapter calls `merlin_commit_state` inside its step timer.
It copies 7,208,960 bytes of KV state and 176 bytes of lengths into separate
static inputs. The prefill-to-decode handoff also copies these tensors once.

This measurement links the **actual compiled runtime, libc, console and startup
objects** from the cached decoder, with their original compilation recipe.
A separate caller supplies synthetic full-size raw state bytes. It checks every
destination byte, source byte and exterior canary outside the timer. It changes
no generated session code and grants no aliasing or pointer-rotation permission.

| Timed work | Cycles |
|---|---:|
| One complete state copy, each of three trials | 4,506,607 |
| 32 complete state copies | 144,210,804 |

The latter is approximately **0.2581%** of the completed 55,871,481,378-cycle
cached session. It is an isolated attribution with synthetic state contents,
not a whole-model optimization or an exact subtraction from session time.
The selected profile, Spike extension and `both/e4m3_vpu/mem.host_tracking=0`
settings match that cached measurement. The actual runtime object SHA is
`dc5d204f1df8e17593d30234612eb044b7ba99955a2ff3ac33377dd635f3187c`.

```sh
python docs/evidence/state_copy_attribution_20261019/verify.py
```

The standalone stdlib verifier checks all 26 retained files, the actual ELF,
structural admission, original runtime source/object identities, state extents,
complete driver observations and timings. The 28,812-byte archive includes the
caller, actual objects, original recipe/projection, runtime source inputs,
commands, complete logs and image. Toolchains and the simulator extension remain
external. Driver byte checks are native observations, not an independent
instruction-level or lifetime proof.

Reproduce with the model-guide environment and a fresh task-local output:

```sh
python -m tools.measure_session_state_copy_spike \
  --stage-build "$CACHED_DECODE_BUILD" --session-root "$CACHED_SESSION" \
  --program decode --profile "$MX_PROFILE" --riscv-root "$RISCV_ROOT" \
  --extlib "$MX_EXTLIB" --structural-elf-admission \
  --out-dir "$TMPDIR/state-copy-attribution"
```
