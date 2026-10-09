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
and E8M0 scale arrays. This checks operation and schedule structure; matrix
command lowering and numerical parity against the source golden remain open.

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
(funct 8) fields. Four scale-selector packets use funct 26. It checks the
Gemmini software header against the RTL repo's pinned submodule and the
Radiance MMIO skip-bit macro. It leaves configuration, E8M0/LUT writes,
fences, address translation, and C move-out to subsequent lowering stages.
