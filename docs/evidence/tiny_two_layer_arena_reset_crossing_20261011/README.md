# Recurrent arena reset crosses the prior failure step

These small logs compare the same two-layer random-weight TinyLlama linked
session on the reference implementation’s pinned MX Spike extension. Both ELFs use the same
prefill and decode stage objects and 19 MX calls per graph. The new session
entry calls `merlin_arena_reset()` before each step; the previous entry did
not. The [summary](summary.json) binds their ELF and log hashes to the same
captured program and profile hashes.

The [old run](without_reset.log) emitted all result and state digests for
prefill and decode steps 0 and 1, then exhausted its 2 GiB bump arena in
decode step 2. Spike returned process code 0 despite the guest `FATAL:` and
`*** FAILED` markers. The [reset run's valid prefix](with_reset_valid_prefix.log)
matches those first nine digests exactly and emits all three digests for
decode step 2 before entering step 3. This demonstrates that the per-step
reset gets past the earlier exhaustion point without changing the valid
prefix's outputs or recurrent state.

The reset Spike process was still running when this prefix was archived.
There is no guest `DONE`, no completed 32-step inference claim, and no
independent quantized numerical parity in this evidence.
