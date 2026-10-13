# SmolVLA flow operation profile after weight prepacking

The same captured flow stage as the [paired timing run](../smolvla_flow_prepack_spike_20261013/README.md)
was rebuilt with Merlin's `--op-profile` instrumentation and run once on
the pinned MX+VPU Spike model. The instrumented build produced the same three output
digests. It took 6,660,143,623 modeled cycles, 0.0058% above the
uninstrumented prepacked run. The 6,338 top-level operation intervals account
for 6,642,466,424 cycles (99.73% of the run). There was no profiler overflow.

| Owner of profiled operation | Attributed cycles | Share of stage cycles |
| --- | ---: | ---: |
| MX adapter calls | 4,577,236,015 | 68.7% |
| VPU calls | 501,696,761 | 7.5% |
| Other host operations | 1,563,533,648 | 23.5% |

Four dynamic batched attention call signatures account for 2,273,809,174
cycles inside the MX bucket: `[15,50,163,64]`, `[15,50,64,163]`,
`[15,50,113,64]`, and `[15,50,64,113]`. Their adapter currently converts live
f32 matrix operands to FP8, issues the checked MX command schedule, and widens
BF16 output back to f32. The profile measures these calls as a whole; it does
not yet assign their cycles to conversion, command issue, or readout.
The report joins each MX call to its catalog source region using the verified
entry-to-kernel binding and source operation order. The eight instances of
`matmul_7,25,43,61,79,97,115,133` are the largest individual calls; each
takes about 100 million cycles. The join checks the complete 148-call binding
count and rejects a changed call order.

The remaining host operations include 1,025,356,921 cycles in
`linalg.generic` and 532,297,765 in `linalg.transpose`. These are measured
targets for source-graph scheduling and layout work. The extension reports
7,676,544 mesh busy cycles for the same run, so reducing adapter and host
work is the next performance task before comparing this model section to a
matched handwritten attention kernel.
An [exact live quantizer lookup](../smolvla_flow_fast_quant_spike_20261013/README.md)
subsequently cut this stage's uninstrumented cycle count by 50.6%. The owner
totals above describe the earlier C-reference quantizer build.

The [raw Spike log](spike.log), [operation table](op_profile_table.json.gz),
and [summary report](op_profile_report.json) are retained. Build a diagnostic
stage with `tools.build_merlin_stage --op-profile`, run the resulting ELF with
the same Spike settings, then call `tools.report_stage_profile` to reproduce
the owner totals. Use the uninstrumented build for headline timing.
