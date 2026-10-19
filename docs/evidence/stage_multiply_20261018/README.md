# Whole-stage private multiply handoffs

The stage builder now connects source-verified f32 multiply pairs under the
already selected BF16 VPU policy. The generic Merlin analysis finds private
SSA intermediates and checks intervening tensor work. The MX adapter verifies
the actual arithmetic, uses the existing connected-dispatch realization check,
emits a ranked three-input/one-output ABI, and keeps the BF16 intermediate in
scratchpad. Both arithmetic rounding points remain. Nonfinite and extreme
inputs use the original materialized kernels.

| Executed case | Materialized cycles | Connected cycles | Result |
| --- | ---: | ---: | --- |
| Whole captured 32×32 graph, including an intervening MX projection | 225,327 | 208,477 | 7.48% fewer cycles; all 1,024 f32 output words identical |
| Saved TinyLlama MLP section, 1×64×5632, warm | 22,759,408–22,760,183 | 16,878,353 | 25.84% fewer cycles; every output bit checked |
| Independent 137-element tail, warm | 9,605 | 6,996 | Every output bit checked |
| Extreme-input 137-element fallback, warm | 9,605 | 15,524 | Original path runs; guard overhead is visible |

The small graph is `(a*b) * up(x)`. Its projection originally lies between
the multiplies. Both variants are captured by model2MLIR, compiled by Merlin
through the ordinary MX stage builder, linked, and executed on Spike. The
optimized graph retains two VPU arithmetic commands and removes one
intermediate store/reload (4,096 device DMA bytes). Accelerator load bytes
decrease from 10,240 to 8,192. Catalog and rewrite pair records agree exactly.

The section executes the generated ranked ABI, including its range guard and
output widening. Inputs and the independent scalar BF16 reference are the
existing [checkpoint-backed section archive](../connected_model_section_20261016/README.md).
The section removes 1,441,792 intermediate device DMA bytes per invocation.
Its payload is reused from that archive without another weight or tensor copy.

Both complete TinyLlama source stages also pass the policy-selected VPU rewrite
and typed-module verification. Each has 22 connected MLP pairs: shapes
1×128×5632 for prefill and 1×1×5632 for decode. Those checks do **not** execute
the 22-layer model. No whole-TinyLlama speedup or numerical improvement is
claimed. Softmax and general elementwise additions remain on the host in the
explicit generated policy. The original full-model numerical gap remains open.

All execution uses the same archived `e4m3_vpu` Spike extension as the earlier
model comparison (`5058ebfd…cdc5`), `GEMMINI_MODE=both`, and
`mem.host_tracking=0`. Profile and executable hashes are recorded. This run
uses the archived profile and does not reverify a live RTL source checkout.
The full-stage builds explicitly select Merlin's existing structural ELF
admission service, which records final bytes and declared ISA; it imposes no
accelerator instruction policy. These are modeled Spike measurements, not
FPGA timings or a complete hardware/memory-latency qualification.

The option is experimental and disabled by default. The planner selects legal
candidates for measurement without assuming positive savings on arbitrary
inputs. The fallback result explains why calibration still matters.

## Verify and reproduce

Run `python docs/evidence/stage_multiply_20261018/verify.py` with NumPy. It checks
the manifest, all small-graph output words, source/catalog pair agreement,
section payloads against an independent BF16 arithmetic calculation, output
hashes, and measured transfer savings. It does not reopen omitted ELFs, large
checkpoint files, or full-model source tensor payloads.

With the existing model2MLIR, Merlin and LLVM/RV64 environment from the
[model guide](../../compile_e2e_models.md), create the small fixture:

```sh
python docs/evidence/stage_multiply_20261018/prepare_fixture.py \
  --out "$TMPDIR/multiply-fixture" --profile "$MX_PROFILE"
```

Build that capture twice with `tools.build_merlin_stage`, distinct workdirs,
`--execution-policy "$TMPDIR/multiply-fixture/policy.json"`,
`--structural-elf-admission`, and the same Spike extension. Add
`--connect-vpu-multiplies` to one build. Set the timing environment above and
`--output-dump-cap 1024` to retain every output word.

For a saved section, `tools.qualify_stage_multiply_spike` accepts `--capture`,
`--profile`, `--riscv-root`, `--extlib`, `--startup-dir`, and a fresh `--out-dir`.
The capture requires `source.mlir`, `inputs.npz`, `bf16_policy_golden.npy`,
and a receipt binding their SHA256 hashes. Use `--expect-fallback` only for
the extreme-input fixture. Its receipt records generated sources and binaries,
the extension/toolchain identities, the selected source files, and all three
cold/warm output checks.
