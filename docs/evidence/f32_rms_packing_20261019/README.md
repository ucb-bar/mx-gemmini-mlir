# FP32 RMS normalization and FP8 packing

This paired probe processes 160 rows of width 2048. The control materializes
FP32 square, reduction and normalized tensors, then calls the current
specialized FP8 packer with integer group preparation. The candidate packs
one normalized row at a time using the factory added in MX revision
`272afe6d`. Both retain FP32 normalization arithmetic and canonical BF16-input
FP8 codes with E8M0 block32 scales.

| FP32 sum order | Control cycles | Fused cycles | Ratio | Cycle reduction |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 24,353,720 | 17,440,707 | 1.396× | 28.39% |
| Sixteen lanes with paired elements | 24,304,460 | 17,377,511 | 1.399× | 28.50% |

Each value repeats in two opposite call orders. Every call checks all
337,920 packed bytes and output/workspace guards. The input includes zero,
negative zero, small and large finite rows. A separate scalar FP32 reference
and the retained original scalar quantizer generate the expected bytes.
The reader also checks every byte printed by the target and reopens the
actual ELF's input, gamma and expected-output payloads.

The [complete model execution](../radiance_rocket_optimization_20261019/README.md)
measures **6,001,616,063 → 5,970,974,009 cycles**, a **0.51%** reduction for
45 sequential FP32 RMS fusions, with every logit word preserved. The model's
original LLVM lowering already removes much of the overhead retained by this
materialized C control. The probe's 28% reduction does not predict the
whole-model gain.

These are normalization and packing probe results, not complete model
timings. The sixteen-lane choice changes summation order relative to the
sequential choice and requires separate model quality validation. An earlier
diagnostic used the original generic scalar packer as its timed control;
that larger saving also included packing specialization and is not the
fusion comparison reported here.

To verify the retained observations, choose fresh paths:

```sh
python docs/evidence/f32_rms_packing_20261019/verify.py \
  --workdir "$TASK_DIR/.tmp/f32-rms-check" \
  --out "$TASK_DIR/.tmp/f32-rms-check.json"
```

The reader needs NumPy and a host C compiler. It checks the archive,
structural admission, actual ELF payloads, the independent reference, all
observed packed bytes and the cycle records. The archive retains generated
C, the production and reference packers, constants, the executable, original
build/run commands, target output and timing counters. Toolchain and
extension hashes appear in `artifacts.json`; this is not a hermetic toolchain
archive. The target and extension match the
[complete-forward control](../radiance_rocket_forward_20261019/README.md).

The measurements use modeled Spike cycles with host memory tracking disabled.
The normalization runs on Rocket FP32 instructions and launches no matrix or
VPU operations. They are not FireSim measurements.
