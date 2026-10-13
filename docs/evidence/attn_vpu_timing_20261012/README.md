# MX+VPU attention timing on Nicolas's Spike

The unmodified [handwritten attention program][source]
and the [typed compiler program][compiled]
both pass their output checks with Nicolas's functional plus timing model.
Both runs use the `e4m3_vpu` preset, default timing parameters, Spike SHA-256
`de7573f938268f77b8a1729420d3c47aacebed571efc090da73ee87078ce01e9`,
and `libgemmini.so` SHA-256
`5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5`.

| Program | Modeled cycles | MX commands | Fences | Output check |
| --- | ---: | ---: | ---: | --- |
| Handwritten | 43,641 | 1,079 | 32 | Passed |
| Typed compiler | 51,309 | 1,405 | 34 | Passed |

The compiled program takes 7,668 more modeled cycles (17.6%). Its report also
shows more store traffic (12,288 versus 7,168 Puts) and scale traffic (4,608
versus 2,304 bytes). These differences identify useful scheduling work; they
do not by themselves establish a single cause. The [source log](source_spike.log),
[source model report](source_perf.txt), [compiled log](compiled_spike.log),
and [compiled model report](compiled_perf.txt) contain the complete outputs.

Run either checked-in ELF with the pinned extension and the same environment:

```sh
GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu \
GEMMINI_PERF_OUT="$OUT_DIR/gemmini_perf.txt" \
"$SPIKE" --isa=rv64gc_zicntr --extlib="$SPIKE_EXTLIB" \
  --extension=gemmini "$ELF"
```

The timing model supports compiler iteration. Its host and VPU parameters
still need hardware calibration before treating these numbers as measured
SoC latency. This is a paired kernel comparison, not a full-model result.

[source]: ../nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb/README.md
[compiled]: ../nicolas_attn_vpu_fa_typed_program_a042643_0344bcb/README.md
