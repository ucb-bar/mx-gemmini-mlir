# Resident sigmoid and multiply chain

This experimental emitter retains sigmoid and two following multiplies in
scratchpad under the existing BF16 VPU policy. It preserves sigmoid's four
original VPU instructions and both multiply instructions, including their
rounding. Merlin checks the five private physical handoffs. The provider checks
the actual strict source arithmetic and private source graph; names alone cannot
qualify a changed sigmoid. The feature is not yet integrated into whole-model
catalog rewriting or enabled automatically.

| Complete section | Original three calls | Existing multiply-pair connection | Sigmoid plus multiplies |
| --- | ---: | ---: | ---: |
| Saved real MLP operands, 1×64×5632 | 29,904,196 | 24,022,860 | 12,376,328 |
| Independent137-element tail | 12,648 | 9,881 | 5,182 |
| All33,538 BF16 gate encodings with magnitude≤16, varied finite second operand | 3,387,012 | 2,638,322 | 1,353,249 |
| 8,311 elements, last gate20 triggers fallback | 764,873 | 620,827 | 1,066,268 |

Values are the minimum of two warm modeled-cycle measurements after a cold
call. Every retained output word matches the original materialized program;
both optimized variants also check every word against that captured device
output inside each cold/warm execution. Inputs are checked unchanged. No FP32
accuracy gate or whole-model improvement follows from these section comparisons.
The real operands came from a saved source-FP32 checkpoint section, not a dump
of the current quantized full model's internal tensors.

The real section saves48.48% versus the existing multiply-pair optimization.
Across its three calls, extension load bytes fall10,813,440→8,650,752→6,488,064;
VPU command counts remain792 in every variant. The connected emitter needs at
most212,992 bytes of scratchpad, checked against the profile. Original fallback
entry buffers remain in the image; reduced traffic does not establish reduced
reserved host storage. Setup, packing, guards, device work and widening are
inside each measured call. Console output and harness setup are outside.

The guard admits gate magnitudes≤16 and finite second operands under the
existing conservative multiply bound. Other inputs execute the original calls.
The late-fallback fixture is substantially slower: earlier chunks have already
executed before the guard falls back to the full original program. This case
is unprofitable. Automatic selection is absent; improving fallback cost and
integrating the source rewrite remain required work.

All runs use the pinned extension, `both`, `e4m3_vpu`, `mem.host_tracking=0`.
`executions.tar.xz` retains all12 admitted section ELFs, full first-call words,
native drivers/checks, inputs, source/outlined IR, counters and receipts. The
archive verifier reopens every retained word, image/admission digest and three
timing rows. Its structural admission checks image/declared-ISA custody only.

To reproduce from the configured task worktree, capture a fresh fixture with
`tools.capture_sigmoid_multiply_probe` (`--elements 137`,
`--exhaustive-bf16-gate`, or `--elements 8311 --fallback`). Real operands use
`--section-inputs SAVED_MLP_INPUTS_NPZ` with gate in0 and up in2. Then run:

```sh
python -m tools.qualify_sigmoid_multiply_spike --capture "$CAPTURE" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --extlib "$PINNED_EXTENSION" \
  --startup-dir "$BAREMETAL_BENCHMARK_COMMON" --out-dir "$FRESH_OUTPUT" \
  --structural-elf-admission
```

All generated products belong inside the task. The qualifier preserves source capture hashes, full outputs,
selected tool hashes, target profile and startup-source identities.
