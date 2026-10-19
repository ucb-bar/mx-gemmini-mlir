<a id="mxvpu-attention-timing-on-nicolass-spike"></a>

# MX+VPU attention timing on the pinned Spike model

The unmodified [handwritten attention program][source]
and the [typed compiler program][compiled]
both pass their output checks with the reference implementation’s functional plus timing model.
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
versus 2,304 bytes). The [source log](source_spike.log),
[source model report](source_perf.txt), [compiled log](compiled_spike.log),
and [compiled model report](compiled_perf.txt) contain the complete outputs.

The two wrappers do different work in their timed second pass. The reference implementation’s
source loads Q/K/V scales once before both passes and reads score,
probability, and probability codes only in the checking first pass. The
compiler object currently reloads the scales and reads all three diagnostic
buffers in both passes. This explains the direction of the scale and store
traffic difference: the extra second-pass readouts account for 5,120 Puts
and 323 commands, while the repeated scale setup adds three commands and
2,304 scale bytes. Together they account for all 326 extra commands in the
compiled report. The 17.6% figure is therefore a measured difference
between these two complete programs, not an isolated matrix/VPU scheduler
gap. An equal-work pair is needed before attributing the remaining cycles to
command order or dataflow. The [equal-work replay](../attn_vpu_equal_work_20261013/README.md)
uses the same model, source data, and checks with three typed issue phases.
It matches the source's command, fence, and traffic counts and measures
43,523 compiler cycles versus 43,641 handwritten cycles.

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
