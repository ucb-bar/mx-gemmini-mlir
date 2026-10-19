# Compact BF16 producer and activation packing endpoints

These are paired sections on the pinned Spike model. They establish qualified
packing and producer endpoints for a future private VPU-to-matrix handoff.
The whole-network graph pass is **not connected yet**, and no new full-model
speedup is claimed from these measurements.

| Section | F32 boundary cycles | BF16 boundary cycles |
|---|---:|---:|
| Pack 160×2048 FP8 | 13,091,028–13,091,029 | 11,472,914–11,472,915 |
| Connected RMS producer + packing, 160×2048 FP8, warm | 26,729,552–26,730,583 | 24,783,798–24,784,880 |
| Pack 32×96 FP4 | 122,785 | 107,606 |
| Late refusal, pack 32×96 FP4 | 106,303 | 93,140 |
| Fallback multiply producer + packing, 32×256 FP8, warm | 1,173,775–1,176,869 | 1,125,791–1,125,842 |
| Two-chunk multiply producer + packing, 32×288 FP4, warm | 772,020–773,074 | 717,270–717,293 |

Each native program alternates order over three repetitions. The 160×2048
packing-only and producer pairs also have reverse-initial-order programs.
Their first-issued producer observations are 26,735,069 cycles for F32 and
24,789,315 for BF16 in separate fresh programs. Warm means repetitions 1 and 2;
this is not a general reset/cache certification. The timer covers either packing
alone or the synchronous connected producer plus packing, as selected in the
receipt. Allocation initialization, console output and all-byte checking are
outside both timers. Whole-model timing boundaries remain unchanged.

Merlin's word-group emitter now accepts explicit float or uint16_t source
storage with the same ordered conversion/refusal/publication contract. MX's
`MX_QUANT_BF16_INPUT` build adds `mx_pack_activation_bf16`. It reads existing
BF16 words, checks their finite membership, and uses the same scale selection,
codebook, FP4 nibble layout and complete-group publication as the selected F32
packing path. The original conservative shape/span bounds remain. It requires
a complete physically padded matrix; implicit-zero rows and logical column
padding are not newly supplied by this endpoint.

The connected multiply emitter also supplies an explicit BF16 output variant.
Inputs and fallback middle storage remain F32. The required intermediate BF16
rounding, range guards, fallback order and VPU instructions remain in force.
Its final output stores the readback words directly; the alternative removes
the F32 widening and following pack-time narrowing. All published default
endpoints retain F32. Normal, fallback and multi-chunk producer words are
checked against independently captured two-call BF16 arithmetic. NaN input
trap qualification for the new producer endpoint and graph integration remain
required before enabling a whole-network route.

Host tests compare every finite BF16 encoding as its own group maximum, mixed
groups, every nonfinite encoding, signed zero, subnormals, mutated input,
invalid extents, refusal partial writes and output canaries. They compare the
compact route against both selected F32 packing and the independent reference
codebook search. Eight executed native ELF images additionally check every
code/scale byte against a reference, producer words where present, and canaries
on all six calls per program. Native refusal here is a packing return of -1,
with identical earlier writes, rather than producer trap qualification.

The default quantizer object remains byte-identical at 14,496 bytes, SHA-256
`f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0`.
The new selected word-group object is 12,656 bytes; adding the BF16 endpoint
makes it 14,600 bytes. Compiler-reported static stack for the compact activation
packer is 208 bytes versus 224 for the selected F32 activation packer. These
are function frame estimates, not measured peak stack. The shared group-code
refactor changes the selected F32 object identity, so these pairs use both
routes from the same new object. They do not assert machine-code identity to
older full-model word-packing builds. Those running images remain frozen.

Producer programs contain both alternatives, including separate fallback
middle buffers. Their combined ELF size is not a per-variant memory comparison.
Each compact producer output uses two bytes per element rather than four;
whole-network peak allocation savings remain unmeasured. Whole-process device
counters are retained. Packing-only programs issue no device commands; producer
programs retain synchronous VPU execution. Host-memory tracking is off.

Merlin also provides a separate checked cleanup of unused static shape-only
views before private-edge analysis. In the current full-forward IR it removes
132 dead views and exposes 44 RMS-to-preparation edges. It preserves calls,
allocations, unknown/dynamic views and actual fanout. This is an IR opportunity
inventory, not proof of endpoint equivalence or profitability. Later lowering
may already remove those views; no standalone DCE performance gain is claimed.

```sh
python docs/evidence/bf16_activation_pack_20261019/verify.py
```

The verifier uses Python, NumPy and two digest-pinned adjacent independent
readers. It imports no compiler code, reconstructs all packed bytes and producer
goldens, reopens every payload embedded in the actual executed ELF, checks the
complete native records/order/hashes/canaries and admission receipts, and checks
retained object/frame observations. `executions.tar.xz` includes exact native
sources, compiled objects/images, inputs/goldens, counters/logs, commands and
environment. `artifacts.json` pins every retained member. Transitive startup
headers, complete Python dependency closure and toolchains remain external;
the archive is not a hermetic rebuild or independent instruction-policy proof.

Reproduce a packing pair from MX in a fresh task-local directory:

```sh
python -m tools.qualify_bf16_activation_pack_spike \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools \
  --extlib docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so \
  --startup-dir /path/to/gemmini-rocc-tests/riscv-tests/benchmarks/common \
  --out-dir "$TMPDIR/bf16-packing-repeat" --rows 160 --cols 2048 --precision 8 \
  --structural-elf-admission
```

Add `--candidate-first` in another fresh directory. To include the connected
producer, supply `--producer-capture` with a checked
`tools.capture_broadcast_multiply_probe` capture whose result has the same flat
extent as rows×cols. The retained source model, input NPZ, BF16 policy golden,
producer receipt and exact build commands identify each original experiment.
