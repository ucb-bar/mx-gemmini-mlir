# Compiler optimization probes

These are standalone Rocket MX+VPU measurements on synthetic inputs. They
measure compiler changes, using the same target profile and timing settings
as the [full-model comparison](../../fair_full_model_comparisons.md).
They do not establish a full-model speedup over `radiance-kernels`.

| Change | Shape | Control cycles | Candidate cycles | Control / candidate |
| --- | --- | ---: | ---: | ---: |
| Specialized FP8 group packing | 160 × 5632 | 42,042,029 | 35,594,011 | 1.181× |
| Streamed BF16 RMS → FP8 | 160 × 2048 | 34,158,568; 34,151,034 | 15,929,241; 15,925,919 | 2.144× |
| Shared QKV completion | M=160, K=2048, N=2048/256/256 | 5,802,053; 5,801,251 | 5,798,835; 5,798,036 | 1.001× |
| Serial hybrid softmax → FP8 → matrix | 32 matrices, M=160, K=160, N=64 | 70,020,417; 70,015,933 | 69,716,467; 69,711,710 | 1.004× |
| FP32 sigmoid normalization | 160 × 5632 | 17,864,269; 17,861,797 | 22,930,755; 22,928,283 | 0.779× |

Ratios use arithmetic means of the retained trials. FP8 packing ran four
trials with alternating order; its cycles were identical in every trial.
Other paired probes ran twice, reversing execution order. The QKV gain is
about 0.055%, and the softmax/matrix gain is about 0.434%. Neither explains
a large end-to-end speedup.

The hybrid softmax alone took **37,808,474 cycles** for 5120 rows of width
160. That run has no separately timed control in this archive. The sigmoid
alternative changes intermediate precision and costs about 28% more in its
standalone test. It is an explicit accuracy option, not a performance win.

## What changed

FP8 packing moves format selection out of the element loop. RMS fusion
retains the original BF16 arithmetic while avoiding intermediate transfers
and repeated input preparation. Projection grouping retains separate weights
and tile partitions and shares a completion fence; it does not reduce input
load traffic.

Softmax fusion preserves hybrid softmax arithmetic and immediately consumes
one packed matrix at a time. Specializing the packing strides and using the
canonical unrolled word preparation made this version slightly profitable.
The archive also retains three slower versions:

| Version | Control cycles | Candidate cycles |
| --- | ---: | ---: |
| Materialize the full packed batch | 70,020,558; 70,018,036 | 71,721,459; 71,696,197 |
| Serial matrices, generic row packer | 70,020,417; 70,015,933 | 71,651,938; 71,647,181 |
| Serial matrices, fixed strides, looped word preparation | 70,020,417; 70,015,933 | 70,690,099; 70,685,342 |

Selection flags and legality conditions are documented in
[Configure target and precision](../../configure_target_and_precision.md).

## Check the artifacts

`probes.tar.xz` contains the actual linked ELFs, emitted C, input and expected
binaries, original compiler/simulator commands, target logs, performance
counters and structural ELF admission reports. `artifacts.json` pins every
member and the simulator, extension, profile and compiler binaries.

The reader requires Python, NumPy and a host C compiler. Choose a new work
directory inside your task:

```sh
python verify.py --workdir "$TASK_DIR/.tmp/optimization-probes" \
  --out "$TASK_DIR/.tmp/optimization-probes.json"
```

The reader checks all archived hashes and ELF digests, regenerates numerical
oracles with the pinned target VPU reference and the canonical scalar MX
quantizer, and parses the retained observations:

- FP8 packing and RMS check every packed byte and surrounding guard on the
  target. Their logs report complete checks; they do not dump raw outputs.
  The reader regenerates their expected bytes and verifies that the actual
  ELF embeds those bytes and inputs.
- Softmax and sigmoid dump every FP32 output word. The reader compares every
  word against the independent target reference, including both sigmoid
  arithmetic variants.
- QKV grouping dumps every output word. The reader compares them against
  an exact power-of-two matrix oracle.
- Each softmax/matrix probe dumps all 844,800 packed bytes and all 327,680
  matrix words per invocation. Packed bytes match the independent numerical
  oracle. Every matrix word matches the actual unfused control.

Guards, output checks and printing execute after the timed interval. The
serial softmax probes also validate a standalone producer after timing.
Consequently, their total `perf.txt` counters include verification work and
must not be presented as counters for the timed sequence alone.

Retained ELFs can be rerun directly with the pinned simulator and extension:

```sh
GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu \
GEMMINI_PERF_SET=mem.host_tracking=0 \
  "$SPIKE" --isa=rv64gc_zicntr --extlib="$GEMMINI_EXT" \
  --extension=gemmini -m0x80000000:0x1c0000000 \
  "$TASK_DIR/.tmp/optimization-probes/rms_pack/pair.elf"
```

These are modeled cycles with host tracking disabled, not FireSim or RTL
measurements. Structural admission binds an ELF digest and declared ISA;
it does not prove instruction policy, numerical correctness or timing.
