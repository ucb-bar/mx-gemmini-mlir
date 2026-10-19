# Broadcast multiply residency and call-local preparation

The opt-in multiply connection now handles actual singleton projection maps
and either consumer operand order. Source arithmetic, both BF16 rounding
boundaries, single-use privacy, source placement and effect checks remain
required. The target adapter places the guarded private intermediate on the
supported scratchpad port; exceptional inputs execute the original kernels.
Mixed-rank argument descriptors use Merlin's checked logical ABI emitter.

A second explicit option prepares repeated broadcast inputs once per call.
Merlin plans dense reuse and aligned private storage; MX supplies BF16 RNE,
range checks and its scratchpad implementation. Every invocation refreshes
the prepared values. No cross-invocation pointer cache is used.

| Executed section | Original warm cycles | Connected | Connected with preparation |
| --- | ---: | ---: | ---: |
| 1×160×2048, row and channel broadcasts | 22,984,036–22,984,811 | 17,310,024 | 11,442,731 |
| 1×3×137, partial row | 31,255 | 22,712 | 17,298 |
| 1×1×2048 | 127,477 | 76,001 | 73,954 |

The prepared 160×2048 section uses 50.2144% fewer cycles than the original
two calls and 33.9% fewer than the first connected version. All 327,680 words
are checked against an independently rounded reference on each of three
invocations. Signed zeros, subnormals and narrowing ties occur in the fixtures.
A separate finite-input case verifies final overflow to infinity.

Fallback costs remain visible. An out-of-range channel weight after one
chunk costs 2,944,114–2,946,608 cycles versus 2,019,391–2,020,166 originally.
Preparing the broadcast weight catches that refusal before device work and
reduces it to 2,134,551–2,135,326, still a regression. A late out-of-range full
activation remains slower (3,162,866–3,165,579 cycles), as does an infinite
input. Every completed fallback output is exact. Runtime frequency and
whole-model profitability require separate measurement; both options remain
experimental and off by default.

NaNs in either a broadcast weight or a late full activation retain the original
refusal. The verifier reopens both actual ELFs at the observed trap PCs and
checks their breakpoint instructions. The first observer incorrectly expected
an illegal-instruction trap; this GCC emits `ebreak`. Its failed observer logs
and source are retained under `protocol_correction/`. The corrected observer
requires the actual breakpoint cause and decoded instruction. No output or
performance qualification is inferred from a refusal.

For the 160×2048 section, preparation removes 653,152 repeated host conversions
and adds 4,416 bytes of static storage to the connected executable. Accelerator
load traffic is unchanged by preparation: both connected variants load
5,898,240 bytes across three invocations, versus 7,864,320 originally. The
resident handoff removes 1,310,720 intermediate DMA bytes per invocation.
The complete connected capsule has a larger BSS than the original capsule,
including its private fallback intermediate; `resources.json` reports actual
ELF section extents. These are static resources, not measured peak memory.
The current planner charges alignment to its budget. Earlier section receipts
report payload bytes; `semantic_check_current.json` records both quantities
and confirms every retained kernel C file matches the current emitter exactly.

The 3.59MB archive retains all 26 section ELFs, objects, native logs, counters,
complete embedded input/reference payloads, generated kernels and ABI shims,
compiler source snapshots, invocation records and the selected target profile.
Execution uses the pinned E4M3 VPU Spike extension, `GEMMINI_MODE=both`,
`GEMMINI_PERF_CONFIG=e4m3_vpu` and `mem.host_tracking=0`. Host instruction,
packing, guard, fallback and conversion costs are inside each section timer.
CPU memory-traffic tracking is disabled. This archive does not reverify live
RTL or establish a transitive compiler/toolchain or hardware qualification.

Run `python docs/evidence/broadcast_multiply_20261019/verify.py` with NumPy.
It reopens every member, independently recomputes BF16 references and hashes,
checks the payload embedded in each actual ELF, validates all native output
checks and traffic deltas, and rederives static storage extents and trap words.

To reproduce, run `tools.capture_broadcast_multiply_probe` with `--rows 160
--width 2048 --out-dir "$TMPDIR/broadcast-capture"`, then run
`tools.qualify_stage_multiply_spike` with that capture, fresh `--out-dir`, and
the pinned `--profile`, `--riscv-root`, `--extlib` and `--startup-dir` from the
retained invocation. Add `--prepare-broadcasts` for the prepared variant.
All fixture and execution commands are retained in the archive.

For ordinary model compilation, select `--connect-vpu-multiplies
--connect-vpu-broadcast-multiplies` and optionally
`--prepare-vpu-broadcast-operands`. The aggregate preparation budget is explicit.
The complete TinyLlama source selects 45 normalization pairs in addition to the
existing sigmoid chains. Its matched full-model control and both new candidates
are running separately. These section results do not establish a new whole-model
speedup or numerical-quality result.
