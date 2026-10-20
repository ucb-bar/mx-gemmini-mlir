# Radiance Llama on Rocket MX+VPU

For the recurrent cache port, independent replay and complete-session audit,
see [independent session checks](independent_session_checks.md). The cached
full-model comparison is pending.

`radiance-kernels` now contains a complete Llama application. The source audit
uses revision
[`5866983682583486b38ae64a4d1bfa5e7b2a002e`](https://github.com/ucb-bar/radiance-kernels/tree/5866983682583486b38ae64a4d1bfa5e7b2a002e),
including `kernels/nightly/llama_e2e`. This is newer than the revision used by
the [published kernel controls](fair_full_model_comparisons.md).

The Rocket port executes that application's phase program without compiling
a network graph through Merlin. The generator supplies the model's buffers,
weights, descriptors, and ordering. A small C interpreter calls the ported
kernels at those boundaries. This gives a separate application baseline
from which to measure the compiler's complete-model behavior.

## Source application

Each decoder layer runs residual RMS normalization, a combined QKV matrix,
RoPE and KV-cache preparation, QK, causal softmax, PV, attention-output
quantization and projection, a second residual RMS normalization, a combined
gate/up matrix, SiLU multiplication, and the down projection. Final
normalization and the LM head follow the decoder layers.

The original generator supports one 64- or 128-token chunk over an existing
cache. It does not directly express the published 128-token prefill followed
by 32 individual cached decodes. That session needs an additional state
adapter, including updates to partially filled quantization groups. The
complete SmolVLA trajectory also needs its own source application. A complete
Llama prefill does not establish either session result.

TinyLlama uses 64-element attention heads. For this model, the source's
default attention selection uses separate QK, softmax, and PV phases.
The source's flash-attention path and concurrent fused workers require a
different supported head shape and target features.

## Rocket adaptation

The port is in
[`radiance_rocket_llama.c`](../tools/fixtures/radiance_rocket_llama.c).
[`qualify_radiance_rocket_llama.py`](../tools/qualify_radiance_rocket_llama.py)
checks the selected source files by SHA-256 before generating data or building
an executable. The generated receipt records every adaptation and build
command, source and tool hashes, image admission, execution settings, and
observed output hashes.

The common target is the published `MxE4M3Fp4VpuGemminiRocketConfig`: one
Rocket core, a 16×16 mesh, 256 KiB scratchpad, 64 KiB accumulator, and BF16 VPU.
The original nightly Radiance geometry has a smaller scratchpad and
accumulator, Muon SIMT workers, and a loop-retirement interface.

The port preserves the source phase order, combined QKV and gate/up
matrices, 64-column matrix tiles, K-step partition, and pre-tiled operand
layouts. Native Rocket DRAM loops replace the original split scratchpad
loads and accumulator stores. They retain the mathematical partition but
have a different command stream and overlap behavior. A direct transport-only
probe failed its complete-output check on the pinned Spike extension; its
timings are excluded.

FP32 row arithmetic runs on Rocket. The VPU computes the BF16 exponentials
used by softmax and SiLU. This preserves FP32 normalization and reduction
arithmetic while using the target's available vector exponential. It changes
the exponential approximation from Muon's `fexp.h`. These costs and numerical
differences belong to the port and must appear in any explanation of a
speedup. This port does not establish the fastest possible handwritten Rocket
implementation or reproduce Muon execution time.

Rotary tables retain the source generator's FP64 inverse-frequency, angle
and trigonometric calculations, followed by storage as F32 coefficients.
Their preparation is outside the port's timer. The compiler instead uses
the captured F32 inverse frequencies, F32 angle arithmetic and runtime target
`sinf`/`cosf` calls. Both use the same head width, rotary base and positions,
but their coefficient words can differ. The source-relative numerical gates
cover the complete outputs; the comparison does not claim identical
rotary arithmetic or attribute all preparation savings to graph fusion.

All matrix operands use E4M3 codes and E8M0 scales in groups of 32. The
source-default smoke check retains its FP32-input encoder. The recorded
TinyLlama comparison instead rounds weights and live quantization inputs
through BF16 and clamps the scale exponent to at least −23, matching the
published compiler's packing policy. Host tests compare those code and scale
bytes against the compiler's independent reference encoder, including zero,
signed zero, small values, and rounding boundaries.

## Completed checks

The complete 22-layer, 160-token TinyLlama forward completes in
**7,082,331,012 modeled cycles**. The published compiler control takes
10,326,818,367 cycles, or 45.81% more. This is a raw timing comparison:
the compiler passes the late-33 numerical gate but fails the new all-160
mean-KL gate. Its mean KL is 0.0821650 versus 0.0774637 for the port, a
6.07% increase against the allowed 5%. The compiler has better worst-row KL
and top-token agreement. That fixed-release comparison has no qualified speedup.

The subsequent [optimization measurements](evidence/radiance_rocket_optimization_20261019/README.md)
include a **6,264,147,801-cycle** compiler execution, a **1.131× speedup** over
this port, passing all six numerical limits. It uses FP32 host softmax and
residual state with finite-maximum specialization and RNE underflow shortcuts.
The fixed-release measurement and executable identity above remain unchanged.

The [full-forward archive](evidence/radiance_rocket_forward_20261019/README.md)
retains every output, the floating reference, and both diagnostic profiles.
Profiling preserves all compiler words and all retained port tensors.

Both 64- and 128-token, two-layer source smoke programs complete on the
pinned Rocket-Spike extension. These use random weights and hidden width 256.
All arena guards remain intact. Every byte
matches the source hardware golden through the first layer's normalization
packing, QKV, RoPE, Q/K/V cache layouts, and QK output.

Differences begin at the exponential. For the 64-token check, 154 of 65,536
softmax probability codes differ from the source golden; all 2,048 probability
scale bytes match. These differences propagate through subsequent layers.
The tool retains complete observed tensors and reports the differences.
It does not treat successful execution or finite outputs as a numerical
acceptance result.

A separate reference interpreter replays all 26 phases using the source
layouts, mesh arithmetic, and Rocket VPU exponential. All 32 retained tensor
checks match bit-for-bit at both chunk sizes. The 64-token case also passes
with the compiler's BF16-input packing policy. The
[qualification archive](evidence/radiance_rocket_port_20261019/README.md)
retains the actual executables, complete observations, replay inputs and
expected values, and independent audit reader.

The checker also derives accelerator work from the phase descriptors and
requires the observed native loops, mesh tiles, busy cycles, vector commands,
load bytes, and scale bytes to match. This rejects incomplete modeled work
even when functional outputs are correct.

The source's NumPy model uses idealized matrix multiplication. Its separate
`mx_golden` executable models the reduced-precision mesh arithmetic and is
the reference for exact matrix checks. Agreement with one reference must not
be presented as agreement with the other.

## Reproduce the port

Use the repository revisions, Python environment, RISC-V tools, and profile
from the [model compilation guide](compile_e2e_models.md). Obtain the source
snapshot at the revision above and set these paths:

```sh
export RADIANCE_ROOT=/path/to/radiance-kernels
export RISCV_ROOT=/path/to/riscv-tools
export STARTUP_DIR=/path/to/riscv-tests/benchmarks/common
export PROFILE=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
export EXTLIB=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
```

From the MX repository, run a source smoke check into a fresh output folder:

```sh
python -m tools.qualify_radiance_rocket_llama \
  --radiance-root "$RADIANCE_ROOT" --riscv-root "$RISCV_ROOT" \
  --startup-dir "$STARTUP_DIR" --profile "$PROFILE" --extlib "$EXTLIB" \
  --chunk 64 --phase-timing --out-dir out/radiance-smoke
```

Use `--chunk 128` for the second source-supported chunk size. Phase timing
adds per-kernel counters; omit it for a complete-model timing comparison
against an uninstrumented compiler executable.

Add `--compiler-packing` to exercise the small model with the same live
quantization policy as the real comparison. To replay a completed smoke
program, set `ROCC_ROOT` to `gemmini-rocc-tests` revision
`350547f9843f46f485d4d1dd4c20b2f52f4844bf` and run:

```sh
python -m tools.replay_radiance_rocket_llama \
  --build out/radiance-smoke --radiance-root "$RADIANCE_ROOT" \
  --rocc-root "$ROCC_ROOT" --out-dir out/radiance-smoke-replay
```

For the recorded TinyLlama forward, first export `tinyllama-forward` inputs
using the compilation guide. Point `CAPTURE_DIR` at the exported `capture`
directory, containing the receipt-bound checkpoint, token IDs, and floating
reference:

```sh
python -m tools.qualify_radiance_rocket_llama \
  --radiance-root "$RADIANCE_ROOT" --riscv-root "$RISCV_ROOT" \
  --startup-dir "$STARTUP_DIR" --profile "$PROFILE" --extlib "$EXTLIB" \
  --recorded-forward "$CAPTURE_DIR" --out-dir out/radiance-tinyllama-forward
```

This mode uses all 22 TinyLlama layers, the same 160 token IDs and floating
checkpoint as the published compiler input, and all 160×32,000 logits. The
larger Rocket accumulator admits a 160-row tile. Only the key axis is padded
to 256; the causal mask excludes the 96 extra columns. Their padded matrix
work remains in the source timing. Embedding lookup and final widening to the
model-visible F32 output are inside the timer. Immutable weight preparation, executable
loading, output printing, and validation are outside it on both sides.

The output includes `session.elf`, generated sources and data, `phases.bin`,
the raw simulator log, complete observed tensors, `logits.npy`, and
`receipt.json`. A completed receipt initially says
`completed_requires_matched_comparison`. It reports floating-reference
diagnostics without declaring a full-model speedup. Compare the completed
run with the published compiler control using:

```sh
python -m tools.compare_radiance_rocket_forward \
  --port-build out/radiance-tinyllama-forward --capture "$CAPTURE_DIR" \
  --out out/radiance-tinyllama-comparison.json
```

This reopens the original compiler archive and every source output byte.
It withholds a qualified speedup unless the source-relative numerical gate
passes on both the 33 previously evaluated positions and all 160 positions.
The new 160-position check is possible because this port retains every logit.

Before publishing a ratio, bind the compiler's complete outputs and receipt
to the same inputs, checkpoint, target, simulator, extension, timing settings,
and logical output boundary. Apply the documented numerical acceptance gate
to the same positions. Explain kernel tiling, placement, numerical policy,
padding, and preparation costs separately from transformations enabled by
whole-network visibility.
