# Complete SmolVLA session: shared activation packing

The candidate completes the same prefix, ten flow-denoising steps, and final
action decode in **1.9038% fewer modeled cycles**. All 33 output/state hashes
and all eleven complete published action tensors are identical to the compiler
baseline. This is a complete linked session comparison, including measured
reset and program-state handoffs.

| Measured work | Baseline cycles | Reuse cycles | Reduction |
|---|---:|---:|---:|
| Reset | 1,450,522 | 1,450,522 | 0% |
| Prefix, including preparation | 66,619,956,522 | 66,127,439,053 | 0.7393% |
| Ten flow steps, including preparation | 23,050,307,607 | 21,835,667,692 | 5.2695% |
| Action decode, including preparation | 8,648 | 8,648 | 0% |
| Complete measured execution | 89,671,723,299 | 87,964,565,915 | 1.9038% |

The timing scope is the sum of reset, program-state preparation, and the existing
step compute intervals. Console printing and phase bookkeeping are excluded in
both runs. This is modeled execution time, not simulator wall time. Total measured
instructions decrease from 87,741,834,907 to 86,034,129,849. Preparation is unchanged
at 2,903,484 cycles. Both real simulator processes exited zero and printed the
complete result protocol and `DONE`.

Merlin recognizes pure equivalent BF16-to-F32 widenings as the same preparation
input under checked ownership and consumer conditions. MX consequently prepares
one packed activation for multiple compatible matrix consumers. The candidate
removes 48 preparations across 32 prefix groups and 32 across 24 flow groups;
each flow program executes ten times. The independent numerical placement,
checkpoint packing, native issuers, and timing boundaries are preserved.
The linked receipts identify changes to only `model.o` and
`device_catalog/mx_catalog.o` for the prefix and flow stages. The action stage
is unchanged. The existing section evidence is in
[widening_reuse_20261018](../widening_reuse_20261018/README.md).

Both runs use the same captured IR, inputs, program schedule, checkpoint operand
pool, target profile, Spike binary, extension, ISA and memory arguments.
The shared program pool is 97,736,960 bytes, SHA-256
`47a1042226334c932f80f4e67d134268335b92fd9bdb7e7a50625d65ff02c5f6`.
Prefix storage occupies 97,679,616 bytes; flow storage occupies 57,344 bytes at
that offset; action storage is empty. The admitted linked ELF grows from
483,451,800 to 485,475,120 bytes. Both sessions use a 6 GiB arena at `0x210000000`,
with the weight pool at `0x200000000`. Allocation-site sums in stage receipts
are not peak-memory measurements.

Both whole-process counter reports contain 18,034,641 commands, 446,775 loops,
201,109 fences, 48,141,352 mesh tiles, 770,261,632 mesh busy cycles and 142,409
VPU commands. Loads are 12,580,427,280 bytes in 340,109,841 Gets; scales are
315,957,264 bytes; stores are 81,102,340 reads and 89,721,028 Puts. These counters
cover the process and have a wider boundary than the summed execution intervals.
Host-memory tracking is off. Reported host stalls increase by 547,674 cycles;
cache/DRAM observations also differ. They are retained rather than assumed
equal. This optimization saves repeated host preparation, not native commands.

The captured unquantized FP32 action comparison is diagnostic. Final cosine is
0.9752036640, MAE 0.03336054171, RMSE 0.07782276601, normalized RMSE 0.2398155381,
and maximum absolute error 0.4540897161. All ten step diagnostics are retained.
There is no established SmolVLA action-quality threshold or complete independent
reference session baseline here. The result establishes preservation of the
compiler baseline's full outputs and numerical behavior. It does not establish
new action-quality acceptance. Older SmolVLA archives use different stage objects
and numerical choices; their cycle counts are not controls for this experiment.

```sh
python docs/evidence/smolvla_widening_session_20261019/verify.py
```

The standalone verifier needs Python and NumPy and imports no compiler code.
It independently reads the exact ordered console protocol, reconciles all twelve
step intervals plus reset/preparation, checks all 33 hashes, decodes all eleven
full action dumps, compares both runs byte for byte, recomputes FP32 diagnostics,
checks the original receipt identities, and checks the retained admission reports.
The final flow/action identity uses exact bytes, including signed zero.

`executions.tar.xz` retains both complete logs, counter reports, output binaries,
live process environments, runner commands/status, stage/link receipts, catalog
and rewrite receipts, generated callers including input headers, captured source
IR and input arrays, weight manifests, FP32 goldens, and comparison readers.
`artifacts.json` pins every member and records its original workspace path.
These are measurement and source-custody records. Historical stage quantizer
sources are not reconstructed from current mutable sources.

Large linked ELFs, stage objects, checkpoint tensors, and transitive toolchain
inputs remain external. This compact archive is not a hermetic rebuild or an
offline replay of ELF admission. To rehash the actual two retained linked images:

```sh
python docs/evidence/smolvla_widening_session_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

For reproduction with the original built images and captures, unpack the archive
under a new task-local directory and replay each `reference/job/spec.json` and
`candidate/job/spec.json` command with a fresh `--out-dir`. The commands reference
the original external link receipts and toolchain paths. To relink the retained
completed stage objects into fresh session images, run from MX:

```sh
for variant in baseline widening_reuse; do
  python -m tools.link_merlin_session \
    --session-root "$TMPDIR/captures/smolvla/smolvla_session" \
    --stage-build "prefix_encode=$TMPDIR/mx_model_goal_20261018/smolvla_prefix_$variant" \
    --stage-build "flow_denoise=$TMPDIR/mx_model_goal_20261018/smolvla_flow_$variant" \
    --stage-build "action_decode=$TMPDIR/mx_model_goal_20261018/smolvla_action" \
    --workdir "$TMPDIR/smolvla_relinked_$variant" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
    --merlin-src "$PWD/../merlin/src" --riscv-root /path/to/riscv-tools \
    --structural-elf-admission --measure-preparation --pack-program-weights \
    --dump-output flow_denoise:0 --dump-output action_decode:0 --dump-limit-bytes 16384
done
```

The exact full-stage compilation recipes are retained inside this archive; the
earlier widening archive also provides independent projection-stage reproduction
commands. The full-session jobs use the originally completed model stage objects.
Retain the original profile, checkpoint bytes, captured inputs, dump selection,
arena, timing settings, and numerical policy. Rebuilding with newer compiler
sources is a new experiment rather than reproduction of these exact ELF hashes.
