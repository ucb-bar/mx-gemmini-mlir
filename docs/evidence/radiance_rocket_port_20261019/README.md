# Radiance Llama Rocket port qualification

This archive checks the Rocket adaptation of `radiance-kernels`
[`5866983682583486b38ae64a4d1bfa5e7b2a002e`](https://github.com/ucb-bar/radiance-kernels/tree/5866983682583486b38ae64a4d1bfa5e7b2a002e),
including its complete Llama phase program. The
[port guide](../../radiance_rocket_model_baseline.md) describes the target,
quantization, adaptations, and reproduction commands.

The source's two-layer smoke model uses random weights, hidden width 256,
four query heads, two KV heads, and a 512-token vocabulary. These checks
validate the port; they are not TinyLlama performance results.

| Completed smoke case | Retained tensor checks | Checked bytes | Exact replay |
| --- | ---: | ---: | --- |
| 64 tokens, original source packing | 32 | 1,218,048 | Every bit |
| 128 tokens, original source packing | 32 | 2,368,512 | Every bit |
| 64 tokens, compiler BF16-input packing | 32 | 1,218,048 | Every bit |

Some checks are overlapping views of the same tensor. The counts above do
not imply 32 distinct allocations. They cover the first layer's intermediate
matrices, packed operands and caches, both layers' residual outputs, and
the complete final logits.

The independent replay executes all 26 phases from the original descriptors.
It uses NumPy for row arithmetic and layout conversion, the source's
reduced-precision mesh reference for matrix products, and the pinned Rocket
VPU reference for exponentials. It does not execute the port's C kernels.
Every retained observed byte agrees with this declared arithmetic.

The unmodified source golden also agrees through the first layer's QK result.
Later differences from that golden start at the exponential: the Rocket VPU
replaces Muon's `fexp.h`. In the 64-token case, 154 of 65,536 probability codes
differ while every probability scale byte agrees. The replay verifies the
subsequent computation against the Rocket approximation rather than treating
those differences as unexplained tolerance.

Both executables pass structural image admission, complete-output checks,
and arena guards. The reader independently derives native loop counts, mesh
tiles, modeled busy cycles, VPU commands, and load/scale bytes from each phase
program, then checks the actual simulator counters. These are numerical and
timing-model checks, not FPGA measurements.

## Additional native matrix controls

Seven separate controls run `npu-exploration` native functions at
`0344bcba70b899039ccd0ac0801ca5605d46d3eb` against compiler-generated issuers,
using the historical Radiance fixture operands. These do not execute the
Radiance Muon kernels or establish a full-model comparison.

Each executable runs three paired trials with alternating call order. A
second executable reverses the initial order. Every output word and guard is
checked after every call. Packing and validation are outside both timers;
completion is inside them. The table gives medians of trials 1 and 2 across
both initial orders: four later observations per implementation. First-call
results remain in the logs and [summary](native_pairs_summary.json); there is
no claim that later calls represent a universally warm cache state.

| Fixture | Native cycles | Compiler cycles | Native/compiler |
| --- | ---: | ---: | ---: |
| FP8 WS, 256×64×2048 | 133,820.5 | 140,395 | 0.9532 |
| FP4 down projection, 256×64×5632 | 94,167.5 | 105,011 | 0.8967 |
| FP8 re-stream fixture, 256×64×2048 | 133,820.5 | 140,395 | 0.9532 |
| FP8 GEMV fixture, 32×128×2048 | 33,756.5 | 33,656 | 1.0030 |
| FP8 GEMV fixture, 64×128×2048 | 66,783 | 66,963.5 | 0.9973 |
| FP8 GEMV fixture, 128×128×2048 | 133,535.5 | 134,050.5 | 0.9962 |
| FP4 GEMV fixture, 128×128×2048 | 33,899 | 34,092 | 0.9943 |

The re-stream row uses that fixture's data with the selected native function;
it does not reproduce the original four-pass re-stream schedule. The compiler
is slightly slower in several controls. These measurements do not imply that
whole-network compilation improves every matrix kernel.

## Inspect and reproduce

Run the portable reader with Python and NumPy:

```sh
python docs/evidence/radiance_rocket_port_20261019/verify.py
```

It checks both archive manifests, actual ELF and producer hashes, complete
observed streams, references bound to the generated source data, guards,
replayed expected words, paired timings, and complete accelerator work.
Original toolchain binaries, checkpoint data, and complete dependent source
checkouts remain external build inputs; this is not a hermetic rebuild.

`qualifications.tar.xz` retains the two original-packing smoke runs, all seven
native pairs, pinned source/reference files, and small compatibility probes.
`replays.tar.xz` retains the three exact numerical replays and the actual
compiler-packing smoke executable. Each run includes its C/data sources,
commands, admission, raw logs, counters, and complete observations.

The failed transport-only GEMM probe is retained and excluded from timing
claims. An initial real-model prototype ended its timer at BF16 logits; that
incomplete run was stopped before acceptance. Its abort record is retained.
The corrected model run includes embedding lookup and final F32 widening.
This qualification archive supplies no accepted full-model speedup.
