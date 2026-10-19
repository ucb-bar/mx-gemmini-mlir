# Exact repeated RHS preparation within a synchronous batch

Merlin proves that adjacent batch matrices contain exactly the same source
coordinates through pure singleton broadcasts, linear reshapes and matrix-axis
transposes. The Gemmini adapter then packs the RHS once per proved group rather
than once per matrix. Each call refreshes its first group and every subsequent
group. A packing, matrix invocation order, output conversion, original rounding,
workspace extent and synchronous completion remain unchanged. This retains host
preparation; it does not remove source broadcasts or device transfers.

The optional `--reuse-batch-rhs` stage flag requires explicit numerical policy
and integer-group packing. The target binds the original source, native kernel
family, compiled objects and C sources. Each current call is independently
re-proved, including calls sharing a declaration; unproved calls keep their
original endpoint. Disjoint call sets can have different proved repeat factors.
Model names, captured values and provenance identities do not choose strategy.

Seven native batch comparisons include padding/packing, native execution and
output conversion inside each timer. Normal cases execute three invocations,
changing the first and an interior RHS group. All complete outputs match
independently computed integer matmul goldens. Three refusal cases preserve
every partial-output word and exterior guard and reach an actual ELF `ebreak`:
NaN in the LHS, NaN in an interior RHS group, and a finite F32 value whose BF16
conversion overflows. Both FP4 and FP8 cover simultaneous M/N/K tails.

| Batch/M/N/K; repeat | Format | Cold original cycles | Cold retained cycles | Private workspace, both |
|---|---|---:|---:|---:|
| 32/1/160/64; 8 | FP8, implicit zero rows | 15,111,468 | 2,275,500 | 18,784 B |
| 32/1/64/160; 8 | FP8, implicit zero rows | 15,376,044 | 2,537,777 | 17,296 B |
| 6/19/23/37; 3 | FP8 | 1,078,391 | 728,352 | 24,704 B |
| 6/19/23/37; 3 | FP4 | 1,131,830 | 749,703 | 22,656 B |

A complete captured graph, with random F32 inputs, grouped broadcasting and a
transposed RHS, measures **1,118,895 → 765,994 cycles (31.5401% fewer)**. All
2,622 output F32 words match exactly. Its commands (78), loops (6), fences (246),
VPU commands (0), mesh tiles (96), load bytes (24,576), scale bytes (768) and
store Puts (192) match. Literal allocation sizes sum to 51,528 → 51,568 bytes
across 3 → 8 sites; this is not peak memory. Images are 52,736 → 57,184 bytes.
Private native workspace is unchanged. Wider process counters have host tracking
off. All native pairs also preserve these device work and transfer counters.

These are native sections and a complete small graph, not a measured TinyLlama
speedup. The full forward and cached decoder each prove 44 eligible calls in
two shared declaration families. Their first builds stopped before simulation
on an overly strict one-declaration-per-call check. A tested call-specific fix
was committed in Merlin `ee99a5c22` and MX `eacbcff`; fresh full-model retries
are pending. The frozen native and small-graph images precede that correction
and are retained unchanged. The graph also predates later source-snapshot
metadata; no historical receipt was amended.

```sh
python docs/evidence/repeated_batch_rhs_20261019/verify.py
```

The independent stdlib/NumPy reader reopens 586 files, 16 actual ELFs and 99,296
observed output words. It checks embedded native payloads, all normal goldens,
changed groups across calls, independent partial-output expectations, actual
breakpoint instructions, guards, admissions, resource records, counters, full
graph words, original timing and object/source hashes. It imports only the
pinned ELF reader from the adjacent broadcast archive, not compiler modules.
The 340,684-byte archive includes capture inputs, generated IR/C, objects,
images, commands, logs and implementation snapshots. Toolchain executables,
native startup dependencies and the simulator extension remain external; this
is completed native execution/custody evidence, not a hermetic rebuild.

With the environment in the model guide, reproduce the first native pair in a
fresh task-local directory:

```sh
python -m tools.qualify_batch_rhs_reuse_spike \
  --profile "$MX_PROFILE" --riscv-root "$RISCV_ROOT" \
  --extlib "$MX_EXTLIB" --startup-dir "$MX_STARTUP" \
  --batch 32 --m 1 --n 160 --k 64 --repeat 8 --precision FP8 \
  --virtual-zero-padding --structural-elf-admission \
  --out-dir "$TMPDIR/repeated-rhs-native"
```

Swap N/K for the PV case; use batch6/M19/N23/K37/repeat3 without virtual
padding for the two tail formats. Add `--refusal lhs` or `--refusal rhs` for
NaNs; the overflow case adds `--refusal lhs --refusal-bits 0x7f7fffff`.
Capture the graph with `tools.capture_repeated_batch_matmul`, groups2/repeat3/
M19/N23/K37 and `--transpose-rhs`. Build twice with the same generated policy,
`--integer-quant-groups`, prepacked checkpoint selection, structural admission
and output dump cap2622; add `--reuse-batch-rhs` to the candidate. Exact original
command arrays are archived under `graph/commands/`.
