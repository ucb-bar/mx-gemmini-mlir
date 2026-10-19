# Exact-byte word-group preparation and loop specialization

These are host packing sections, with guards and packing inside each measured
call. They do not establish a whole-model improvement. The selected variant
preserves all output bytes, scale bytes and refusal progress while reducing
measured cycles by 20.1–40.2% across the current tested cases.

The original quantizer already prepared each group into `float[32]`. The change
prepares `uint16_t[32]`, compares unsigned magnitude keys, and specializes the
group loop into its fixed-axis callers. Merlin owns the reusable strided
conversion/maximum loop; MX supplies BF16 rounding, refusal, scale, FP8/FP4
codebooks and packed layout. No cross-call cache or device command is introduced.
`--integer-quant-groups` selects the combination explicitly and defaults off.

| Case | Shape / live rows | Original | Float inline | Words | Words inline |
|---|---:|---:|---:|---:|---:|
| FP8 activation | 16×32 | 33,733 | 26,110 | 27,966 | 20,404 |
| FP4 activation | 32×32 | 68,328 | 50,882 | 52,260 | 40,890 |
| FP8 activation | 160×2048 | 21,620,885 | 16,727,324 | 17,954,552 | 13,091,094 |
| FP4 activation | 160×2048 | 21,897,984 | 16,305,389 | 16,764,095 | 13,103,007 |
| FP8 activation | 160×5632 | 59,447,478 | 45,990,173 | 49,363,306 | 35,988,744 |
| FP8 weight | 64×128 | 523,292 | 402,789 | 407,473 | 318,462 |
| FP4 weight | 64×128 | 529,839 | 424,916 | 417,206 | 339,281 |
| FP8 virtual zero rows | 16×2048 / 1 | 241,672 | 211,008 | 218,728 | 189,105 |
| FP4 virtual zero rows | 32×2048 / 1 | 252,439 | 223,564 | 220,367 | 201,624 |
| FP8 activation, late Inf refusal | 64×128 | 503,097 | 389,047 | 417,625 | 304,585 |
| FP4 weight, late NaN refusal | 64×128 | 511,688 | 410,265 | 402,846 | 327,597 |

All three invocations of each retained variant report the same cycle count.
Refusal occurs at source index 6007; untouched bytes retain their original A5
sentinel, including partially filled FP4 bytes. Virtual rows are encoded as
zeros without materializing a full float input. Separate host differential
tests exercise all 65,280 finite BF16 words, F32 narrowing ties, signed zeros,
subnormals, rounded overflow, NaN/Inf and wrapped-NaN membership.

An earlier OOT prototype is retained as an ablation: for FP8 160×2048, words
alone took 22,470,392 cycles, a 3.9% regression; words plus inlining took
13,429,331. Promoting the preparation loop into Merlin also hoisted its strided
base calculation. The table reports the resulting implementation, whose
machine code differs from that prototype. An in-flight full forward compiled
with the prototype must remain separately attributed.

The prepared group cache shrinks from 128 to 64 source bytes. GCC's actual
RV64 stack reports give selected activation/virtual-row/weight frames of
224/240/240 bytes, compared with original public frames of 96/192/96 plus a
288-byte private group frame. These are individual compiler-reported frames,
not a complete-program peak-stack measurement. The selected object is 12,656
bytes versus 14,496 for the original. No heap allocation is added. The default
object is byte-identical to the earlier full-forward reference object,
SHA256 `f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0`.

`executions.tar.xz` retains 48 actual executed ELFs: four variants for eleven
current cases plus the prototype case. It also retains actual source snapshots,
generated headers, objects, input tensors, reference buffers, native logs,
admission receipts, tool digests, commands, environment and resource reports.
The pinned profile and extension are unchanged. Admission is explicitly
structural (digest and declared ISA), not a complete instruction-policy proof.
Startup and transitive toolchain headers are external inputs; this archive does
not claim a complete hermetic build closure.

Run the independent artifact check with Python and NumPy:

```sh
python docs/evidence/group_word_packing_20261019/verify.py
```

It checks every archived member, reconstructs scale and code bytes using an
independent grid/search implementation, checks refusal progress, reopens the
payload embedded in every actual ELF, and checks all 144 native complete-buffer
comparisons. It reuses the digest-pinned ELF reader in the adjacent broadcast
evidence directory and imports no compiler code. The baseline-object equality
report is retained; the historical full model object itself belongs to its
separate build and is not included here.

Reproduce a section with the existing toolchains and an unused task directory:

```sh
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export MERLIN_READELF=/path/to/llvm-install/bin/llvm-readelf
python -m tools.qualify_quant_group_words_spike \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --riscv-root /path/to/riscv-tools \
  --extlib docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so \
  --startup-dir /path/to/gemmini-rocc-tests/riscv-tests/benchmarks/common \
  --out-dir "$TMPDIR/quant-group-repeat" --rows 160 --cols 2048 \
  --kind activation --precision 8 --structural-elf-admission
```

Complete TinyLlama forwards separately compare the unchanged control, broadcast
connections, prepared broadcasts, and selected packing. They must finish and
pass complete-logit and existing numerical gates before any new whole-model
performance claim. The last completed qualified result remains 13.736B cycles.
