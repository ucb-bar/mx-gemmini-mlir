# Matching Radiance MX GEMM source kernels

`mx_gemmini_support.source_gemm` reads the current `GemmConfig`, data-header
shape, and `mxgemm<C>` call from each handwritten driver. It checks the source
library's scratchpad assertion and K-loop order before deriving A/B double
buffers, C placement, scale upload sizes, K waves, prefetches, accumulation,
and final move-out. The selected MX target profile separately checks precision,
projection, scratchpad capacity, and output mode.

Run the full source ladder audit with the current local source and RTL:

```sh
python -m tools.match_source_gemm \
  --source-root /path/to/radiance-kernels \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini \
  --out /tmp/mx-source-match.json
```

At Radiance kernels revision `94ba7ca8afe213b92fa2428689fa64800c8eeca9`,
the audit sees 33 shape-ladder drivers. Twenty-two have a feasible source
layout; only two have their data headers present in this checkout. Nine of the
other drivers are rejected by the source's C-placement rule. One uses a custom
contention path and one requests multiple output tiles from a helper that
implements only one. All 19 shape-ladder drivers listed in `MU_SRCS` have a
feasible layout, though most require generated data headers before building.

For the feasible FP8 128×128×512 driver, the source's 128 KiB scratchpad puts
C at row 3072, with four K waves and alternating A/B buffers. The current
`MxE4M3Fp4VpuGemminiRocketConfig` has 256 KiB and puts the same output tile
at row 1024. A target compiler must use the selected profile's address; copying
the handwritten row constant would corrupt the intended layout. The selected
VPU profile admits direct E4M3 and FP4 compute; it has no FP6 LUT path, so FP6
drivers require a different MX profile.
All 81 configurations exported from Nicolas's current `gemmini-mx-cleanup`
source have 256 KiB scratchpads. None is an exact 128 KiB hardware match for
the checked-in handwritten library; the trace below applies that library's
command formulas to a separately selected 256 KiB target layout.
The source MMIO header also places alternate E8M0 scale buffers 2 KiB apart.
The selected target has a 16 KiB scale memory split into 4 KiB halves. The
funct-27 2-D DMA plan loads each 512-byte A/B K-wave slice to destination 0 or
4096, matching the target controller's half-selection bit. It skips the
handwritten loop's unused upload after the final K wave.

Capture the feasible source shape from PyTorch with the selected model2MLIR
checkout, then bind its typed MX handoff to the RTL profile:

```sh
python tests/capture_radiance_mx_gemm.py \
  --model2mlir-root /path/to/latest/model2MLIR \
  --mxq-root /path/to/mxq/site-packages \
  --source-root /path/to/radiance-kernels \
  --out /tmp/mx-gemm-capture \
  --mx-opt build/tools/mx-gemmini-opt \
  --radiance-opt /path/to/radiance-opt \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini
```

The capture receipt records the source driver, data header, latest model2MLIR
revision, quantization site, source tile schedule, target placement, and bound
MLIR digest. The captured PyTorch values differ from the handwritten FP8 code
and E8M0 scale arrays. This capture checks operation and schedule structure;
the separate diagnostic below loads the source's actual values.

The checked-in [source ladder audit](evidence/source_gemm_ladder_20261009.json)
records each driver and its source and target layout. The representative
[capture receipt](evidence/model2mlir_radiance_mx_gemm_receipt_20261009.json)
links the [frontend MLIR](evidence/model2mlir_radiance_mx_gemm_source_20261009.mlir),
[quantization manifest](evidence/model2mlir_radiance_mx_gemm_manifest_20261009.json),
[MX handoff](evidence/model2mlir_radiance_mx_gemm_handoff_20261009.mlir), and
[profile-bound MLIR](evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir)
by SHA-256 digest. The receipt is a structural capture, not an executable
matrix-command or numerical-parity result.

For the same bound MLIR, trace the handwritten library's MX loop-FSM packets:

```sh
python -m tools.trace_source_gemm_loops \
  --mlir docs/evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir \
  --driver /path/to/radiance-kernels/kernels/gemm_mxgemmini/mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp \
  --source-root /path/to/radiance-kernels \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini \
  --out /tmp/mx-gemm-loop-trace.json
```

The [checked-in trace](evidence/model2mlir_radiance_mx_gemm_loop_trace_20261009.json)
contains four prefetch packets and four compute packets, each expanded to the
source macro's bounds (funct 9), scratchpad A/B (funct 24), and loop launch
(funct 8) fields. Four scale-selector packets use funct 26. Eight target
funct-27 packets describe the A/B scale uploads for the four waves. It checks the
Gemmini software header against the RTL repo's pinned submodule and the
Radiance MMIO skip-bit macro. Configuration, Muon shared-memory writes,
fences, address translation, and C move-out remain outside this symbolic trace.

The bounded [source-data diagnostic](evidence/source_fp8_128x128x512_20261009.c)
uses that bound MLIR contract, the checked-in FP8 operand, E8M0, and BF16
golden arrays, and the target's scratchpad and scale-buffer addresses. It
serializes four K waves, moves A/B tiles explicitly, uses funct 27 for E8M0
uploads, and checks all 16,384 BF16 outputs. Generate, build, and run it with:

```sh
python -m tools.run_source_fp8_spike \
  --mlir docs/evidence/model2mlir_radiance_mx_gemm_bound_20261009.mlir \
  --capture-receipt docs/evidence/model2mlir_radiance_mx_gemm_receipt_20261009.json \
  --driver /path/to/radiance-kernels/kernels/gemm_mxgemmini/mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp \
  --source-root /path/to/radiance-kernels \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini \
  --riscv-root /path/to/riscv-tools \
  --out-dir /tmp/mx-source-fp8-spike
```

The runner requires Nicolas's pinned `gemmini-rocc-tests`, `libgemmini`, and
nested `riscv-tests/env` submodules. It builds the extension in the output
directory, compiles the generated C to RV64, and runs that ELF on Spike. The
[execution receipt](evidence/source_fp8_128x128x512_spike_20261009.json)
records **zero BF16 mismatches** and the C, ELF, extension, input, profile,
and tool revisions and digests. This qualifies one serial standalone Rocket
diagnostic. The later [payload-bound compiler path](compiled_mx_pipeline.md)
now lowers the typed contraction to a standalone Rocket ELF and matches the
same source golden without including this handwritten header in generated C.
A source-equivalent Muon MMIO issue schedule, mixed Radiance artifact, and RTL
output check remain to be built.

## Generated source FP4 kernel

The Radiance tree tracks the FP4 64×64×128 driver but generates its data header
on demand. The saved [header](evidence/mxgemm.data.fp4.m64n64k128_20261009.h)
was produced from `gen_mxgemm_data.py` and `lib/golden/mx_golden` at the same
source revision. Its SHA-256 is bound in the capture and execution receipts.
To regenerate it without changing the active Radiance checkout:

```sh
git -C /path/to/radiance-kernels worktree add --detach /tmp/radiance-fp4-data 94ba7ca8afe213b92fa2428689fa64800c8eeca9
make -C /tmp/radiance-fp4-data/lib/golden mx_golden
python /tmp/radiance-fp4-data/kernels/gemm_mxgemmini/gen_mxgemm_data.py fp4 64 64 128
```

Use that worktree as `--source-root` and its
`kernels/gemm_mxgemmini/mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp`
as `--driver` in the capture command above. The capture chooses
`examples/fp4-policy.yaml`, so the model2MLIR matmul becomes an MXFP4 typed
contract. The [FP4 capture receipt](evidence/model2mlir_radiance_mx_fp4_receipt_20261009.json)
binds the [source MLIR](evidence/model2mlir_radiance_mx_fp4_source_20261009.mlir),
[handoff](evidence/model2mlir_radiance_mx_fp4_handoff_20261009.mlir), and
[profile-bound MLIR](evidence/model2mlir_radiance_mx_fp4_bound_20261009.mlir).
The [two-wave trace](evidence/model2mlir_radiance_mx_fp4_loop_trace_20261009.json)
contains four E8M0 DMA packets. `tools.run_source_mx_spike` accepts the same
capture, driver, source, profile, RTL, and RISC-V arguments as the FP8 command
above. Its [generated C](evidence/source_fp4_64x64x128_20261009.c) uses the
source's nibble-packed operands and target-derived scratchpad addresses. The
[execution receipt](evidence/source_fp4_64x64x128_spike_20261009.json)
records **zero mismatches across 4,096 BF16 outputs** on the pinned Spike
extension. Two independent builds produced identical C, extension, ELF, and
Spike-log digests.

## Checked-in source FP6 kernel

The checked-in FP6 128×128×2048 header has row-specific A, B, and C LUTs
(`64×3` words each). `source_fp6.py` reads the packed A/B indices, E8M0
scales, LUT banks, and BF16 golden data. It decodes the first two LUT banks and
re-encodes all A/B indices to prove their byte-for-byte layout. The
[payload receipt](evidence/source_fp6_payload_20261009.json) binds these
component digests to the source revision, typed MX contract, and
`MxE3M2OnlyGemminiRocketConfig` profile. That target has 256 KiB of scratchpad,
so C moves from source row 2560 to target row 512.

Capture the FP6 driver with `tests/capture_radiance_mx_gemm.py` as above, using
`--driver kernels/gemm_mxgemmini/mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp`,
`--policy examples/fp6-source-line0-policy.yaml`, and
`--profile profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json`.
The policy uses **only the source's first A/B LUT line** for structural PyTorch
quantization. The separate source payload retains all 64 A/B/C lines and the
actual packed operands; PyTorch values are still different from the source
blobs. The [capture receipt](evidence/model2mlir_radiance_mx_fp6_receipt_20261009.json)
links the [typed handoff](evidence/model2mlir_radiance_mx_fp6_handoff_20261009.mlir)
and [bound MLIR](evidence/model2mlir_radiance_mx_fp6_bound_20261009.mlir).
The [source trace](evidence/model2mlir_radiance_mx_fp6_loop_trace_20261009.json)
has 16 K waves and 112 loop packets, with alternate E8M0 scale buffers.

Run `tools.inspect_source_fp6` with the capture's `--mlir`,
`--capture-receipt`, and `--policy`, plus the same source, driver, profile, and
RTL paths. `tools.run_source_mx_spike` accepts the same arguments as the FP8
diagnostic and builds the FP6 C and RV64 ELF. Its
[execution receipt](evidence/source_fp6_128x128x2048_spike_20261009.json)
records **zero mismatches across 16,384 BF16 outputs** against the checked-in
source golden. The source's own `mx_golden` also reproduced that golden exactly.

The [emitted C](evidence/source_fp6_128x128x2048_20261009.c) serially reloads
FP6 scales into buffer zero. In Nicolas's pinned Spike extension, the LUT
compute branch reads scale offsets without applying the alternate selector;
the [alternating-buffer trial](evidence/source_fp6_alternating_scales_failed_20261009.json)
yielded 16,368 mismatches. The direct
FP8/FP4 branch applies the selector and their diagnostics keep alternation.
This FP6 workaround is only for the standalone Spike numerical diagnostic.
The source trace retains the target's alternate-buffer commands. RTL and
mixed Radiance execution still require separate validation.
