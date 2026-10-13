# Equal-work MX+VPU attention on Nicolas's Spike

The [typed source-bound attention graph](../nicolas_attn_vpu_fa_typed_program_a042643_0344bcb/README.md)
now issues three checked phases: Q/K/V scale setup once, a first pass with
diagnostic readouts, and a timed second pass with only the output readout.
This matches the work done by [Nicolas's source](../nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/README.md)
for the 64×256×128 dense attention shape. The captured model2MLIR graph and
the E4M3 MX+VPU profile remain the inputs to all three typed modules.

| Program | Timed pass, modeled cycles | Whole run, commands | Fences | MX traffic |
| --- | ---: | ---: | ---: | --- |
| Nicolas source | 43,641 | 1,079 | 32 | 147,456 load bytes; 7,168 Puts; 2,304 scale bytes |
| Typed compiler | 43,523 | 1,079 | 32 | 147,456 load bytes; 7,168 Puts; 2,304 scale bytes |

The compiler is 118 modeled cycles faster (0.27%) in this paired run. Both
programs pass the score mesh golden, all four stage hashes, and the output
checker. The [source log](source_spike.log), [compiler log](spike_1.log),
[source timing report](source_perf.txt), and
[compiler timing report](spike_1_perf.txt) contain the exact measurements.
Two compiler runs produced identical logs; the [receipt](receipt.json) records
source, profile, toolchain, extension, object, ELF, and log hashes.

The [setup](attention.program_bound_setup.mlir),
[check](attention.program_bound_check.mlir), and
[timed](attention.program_bound_timed.mlir) modules lower to 3, 699, and
376 commands respectively. The archived [objects](object/) are data free;
the source supplies identical operands and checker code to both programs.

To reproduce, use pinned `npu-exploration` `0344bcb`, Gemmini RTL
`gemmini-mx-cleanup` `266c593` with its checked-out `mxgen` and
`software/gemmini-rocc-tests` submodules, the RV64 toolchain, Nicolas's
`libgemmini.so`, and a freshly built `mx-gemmini-opt`:

```sh
python -m tools.qualify_nicolas_dense_attention \
  --variant fa --equal-work --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --mx-opt "$MX_OPT" \
  --out-dir "$OUT_DIR"
```

This is a functional Spike timing-model result for one attention kernel.
It is not hardware latency or a full-model performance result.
