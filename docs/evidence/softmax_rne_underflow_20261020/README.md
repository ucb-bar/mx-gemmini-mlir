# Scalar exponential underflow probe

Under round-to-nearest, ties to even, `exp(x)` rounds to positive FP32 zero
for every `x <= -104`. The optional host-softmax policy branches around those
calls and retains the original libm call otherwise. Floating status and
`errno` must be unobserved. The separate guarded-libm contract is unchanged.

The Rocket probe compares the emitted scalar function with the actual pinned
libm endpoint on 4,123 inputs, including cutoff neighbors, the subnormal
rounding boundary, negative infinity and extreme finite negatives. Every
returned word matches. It also checks that the target rounding mode is RNE.

| Timed sequence, 5,120 inputs | Modeled cycles per trial |
| --- | ---: |
| Original libm | 482,223 |
| Explicit underflow branch | 282,542 |

Four trials per sequence give the same counts. Half the performance inputs
underflow; the other half are random values between -20 and zero. This is a
41.4% reduction for this scalar loop, with no full-model speedup implied.
Verification and printing occur outside the timed intervals. These are Spike
modeled cycles with host tracking disabled, not FireSim measurements.

`probe.tar.xz` retains the linked ELF, complete inputs, emitted MLIR/LLVM,
driver, original build/run commands and target logs. `artifacts.json` binds
every member and the implementation revisions. The target driver compares
every output word internally and reports its error count; it does not dump
those words. The full-model comparison has a separate complete-word check.

Run `python verify.py` to reopen every archive member, both complete embedded
inputs, the target checks, each timing interval and the strict mathematical
cutoff. This reader does not rerun the ELF or independently prove the library
implementation. The recorded command can rerun it with the simulator and
extension pinned in `comparison.json`.

Use the [host arithmetic configuration](../../configure_target_and_precision.md#select-softmax-arithmetic)
to select this policy for a fresh model build.
