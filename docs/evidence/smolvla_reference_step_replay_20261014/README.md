# SmolVLA flow step on captured FP32 state

This is a one-step numerical diagnostic on Nicolas's `e4m3_vpu` Spike model.
It reuses the **same compiled flow-stage object, MX catalog, checkpoint weight
blob, and profile** as the [full 12-step run](../smolvla_full_action_values_20261014/README.md).
The input is the captured FP32 step-9 state: the step-8 FP32 flow output,
the original captured prefix cache and mask, and the FP32 Euler timestep.
The captured FP32 quality and correctness trajectories are byte-identical.

| Step-9 output compared with captured FP32 | Cosine | Normalized RMSE |
| --- | ---: | ---: |
| Full compiled prefix + ten carried flow steps | 0.054454 | 1.725878 |
| One compiled flow step on captured FP32 state/cache | 0.907116 | 0.436705 |

The replay replaces **both** the earlier compiled flow state and the compiled
prefix cache. It demonstrates that their accumulated differences account for
much of the final error, but does not separate their contributions. The
remaining one-step error is still substantial and does not meet a model
quality threshold. These inputs are synthetic, as declared by the original
capture. The one-step replay reports 41,133,937,106 modeled cycles, but it
starts with cold weight/cache state and excludes the prefix and action stages;
there is no valid cycle ratio against the full session.

The full-run hashes also show that the f32-ABI timestep was computed with
BF16-rounded VPU additions: all ten result hashes match the BF16 recurrence.
At the last step, the compiled timestep is `-0.0048828125`; the FP32 source
recurrence is approximately zero. The VPU catalog binds the captured
one-element `add_96` region to a BF16-rounded f32 VPU add. This is a
separate precision issue; these data do not prove how much it contributes to
the action error.

`prepare.py` builds a one-step diagnostic session using hard links for the
immutable model and weights. It does not modify the original capture. The
derived input and source identities are in `derived_replay_receipt.json`.
`archive.py` copies only the small output, reference, run receipts, and log;
`verify.py` recomputes the artifact hashes, numerical metrics, flow-object
identity, and timestep hash sequence:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B \
  docs/evidence/smolvla_reference_step_replay_20261014/verify.py
```

For a fresh local run, prepare a new task `.tmp` session from the pinned
model2MLIR SmolVLA capture, link `flow_denoise` with the same stage build and
`MxE4M3Fp4VpuGemminiRocketConfig.json`, then run the linked ELF with
`tools.run_linked_merlin_session --perf-config e4m3_vpu
--perf-set mem.host_tracking=0`. The archived link and run receipts give the
exact profile, objects, Spike extension, and input digests.
