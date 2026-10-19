# Guarded native tile search

These are paired encoded-issuer sections, not complete model measurements.
Packing and validation are outside these section timers; dispatch, configuration,
transfers, compute and the existing completion boundary are inside. Full model
timing still includes packing and all other original work.

The existing native selector takes the first legal tile from a short list. For
160-row FP8 projections, that misses the legal 80-row partition. Merlin now
supplies bounded aligned-divisor enumeration and selection under complete
nonincreasing structural-work vectors. MX supplies physical admission and exact
encoded-input/scale traffic and command counts. Its explicit experimental
`--native-tile-policy movement` preserves the original K partition and requires
nonincreasing total input bytes, commands, maximum single-launch input bytes,
and live accumulator footprint. Capacity checks, pointer-span checks, scratchpad
half alternation and the original completion boundary remain in force.

These quantities are structural proxies, not cycle predictions. Unknown work
cannot become zero; inconsistent metrics refuse. Equal vectors and search-budget
exhaustion retain baseline. FP4 retains its original schedule because this
selection family is not yet numerically qualified for that precision. No model
name, provenance ID or golden output selects a strategy. The default is legacy.

The selected large-projection partition changes 160×64×512 to 80×128×512. It
reduces encoded-input and scale reads by 7.14%, with the same native launch count,
accumulator footprint, K accumulation partition and arithmetic tile count.
The independent 192-row case selects 96×128×512 from 192×64×512 and reduces those
reads by 12.5%. These choices come from current shape/capacity facts.

| FP8 shape M,N,K | Baseline warm cycles | Selected warm cycles |
|---|---:|---:|
| 160,256,1024 | 167,400–167,430 | 166,456–166,818 |
| 160,2048,2048 | 2,681,737–2,686,243 | 2,662,091–2,663,878 |
| 192,384,1024 | 301,659–302,850 | 300,358–301,752 |
| 160,5632,2048 | 7,399,606–7,399,714 | 7,336,439–7,338,430 |
| 160,2048,5632 | 7,617,997–7,618,486 | 7,315,268–7,316,370 |

“Warm” here means repetitions 1 and 2 of these fresh simulator programs, with
alternating call order. It is not a general cache/reset certification. The
192-row ranges overlap slightly; all paired observations favored the candidate,
but this is not a robust separated-band speedup. The large 2048² projection was
also run in both initial orders: the first-issued control took 2,686,842 cycles
and the first-issued candidate took 2,663,331 in the reversed program. The gate
case has only baseline-first evidence; the down case has only candidate-first
evidence. Their matched first-issued comparisons remain unmeasured.

The retained counterexamples motivated the guards:

- Attention 160×160×64, proposed 160×80×64: fewer total bytes and launches, but
  the matched first-issued candidate regressed 9,844→11,056 cycles. Warm results
  improved 9,006→8,516–8,548. Per-launch transfer and accumulator footprint grow.
- The 80×80×64 attention variant improved warm execution 9,006→8,001, but its
  matched first-issued result regressed 9,844→10,050. Its accumulator footprint
  grows. The final policy retains the original 160×32×64 partition.
- A physical 16-row 16×1600×512 probe with 16×160×512 tiles looked faster in the
  first paired observation, then regressed warm execution from 83,573–84,198 to
  90,399–92,606. Per-launch input and accumulator footprint grow; it is rejected.
  This is an independent encoded probe, not a cached-session decode measurement.

The attention pairs report 6,240 combined Puts versus 4,800 logical store-read
units, exposing partial-write overhead. Identical logical output bytes do not
imply identical physical transaction counts. The artifact check predicts store
read units; Puts for refused partitions remain an observation, not an invented
transaction model. Accepted pairs have no extra Puts in these observations.
All paired programs report 20 fences in their combined harness scope. The
actual emitted native loop body already batches commands without a per-loop CPU
fence; only its original call completion is emitted. An earlier task note's
claim of a fence after each loop instruction was incorrect.

`executions.tar.xz` retains twelve actual executed ELFs, encoded inputs and scales,
integer goldens, raw logs/counters, structural admission receipts, source/MK and
producer snapshots, tool digests, commands and environment. The payload embedded
in every ELF is independently reopened. Integer cases compare both schedules
against an independent int32 dot product rounded to BF16. Floating cases compare
every BF16 output against the unchanged native schedule; the unused zero golden
is not an independent floating oracle. All three calls per schedule are checked,
with output canaries and hashes. Whole-model numerical gates remain separate.

Combined native command, input-byte and scale-byte counters match the independently
recomputed schedule quantities exactly. Counters include all six issuer calls;
they are not per-variant isolated measurements. Host tracking is off. No peak-stack,
host-traffic, asynchronous-overlap or complete hardware timing claim is made.
Startup and transitive toolchain headers remain external build inputs.

Default C and canonical MK text were checked against the actual prior full-build
sources for all seven unique issuers. The retained copies/digests establish that
source identity, not a new whole-model machine-code identity. The movement policy
changes five issuer families in the independent 160-token catalog; other adapters,
packed operands and numerical placement are unchanged by this selection.
`build_isolation.json` separately records the live builds' identical host model,
host-math, quantizer and packed-weight objects. Only five native C issuer sources
change; the packed-weight assembly directive paths also differ between workdirs.
That report is a workspace observation. These full-model objects/ELFs are not in
this section archive, and its verifier checks the report digest only.

A combined regression run also exposed a native-test lifetime bug: passed-test
cleanup reused a truncated temporary pathname for two different dense libraries.
`dlopen` returned the same handle for different requested binary hashes, causing
the wrong shape-specific kernel to write past its output. The existing dense
tests now keep compiled images in session-owned directories. The original
combined run passes without changing pytest's retention policy. A tagged debug
log is retained as diagnosis evidence; no debug hook is in production/tests.

```sh
python docs/evidence/native_tile_search_20261019/verify.py
```

This needs Python and NumPy and reuses the digest-pinned adjacent ELF reader. It
imports no compiler code, reconstructs independent integer references and all
payloads, checks native comparisons and observed traffic, and checks policy
acceptance/refusal against the four structural quantities. It does not promote
those quantities to a numerical theorem or a cycle forecast.

Reproduce a pair using an unused task directory:

```sh
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export MERLIN_READELF=/path/to/llvm-install/bin/llvm-readelf
python -m tools.qualify_native_tile_pair_spike \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools \
  --extlib docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so \
  --startup-dir /path/to/gemmini-rocc-tests/riscv-tests/benchmarks/common \
  --out-dir "$TMPDIR/native-tile-repeat" --m 160 --n 2048 --k 2048 \
  --tile 80,128,512 --pattern fp8 --structural-elf-admission
```

Use `--candidate-first` in another fresh directory to reverse the initial order.
A separate complete 160-token forward with movement selection is running against
the otherwise identical guarded-math/packing control. Complete logits and the
unchanged numerical gate are required before reporting a new model result.
