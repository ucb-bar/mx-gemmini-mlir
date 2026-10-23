# Complete cached TinyLlama comparison

The independent Radiance-derived Rocket MX+VPU port completes the full
128-token prefill and 32 teacher-forced single-token decodes. The compiler
is slower and fails the predeclared worst-row KL limit. This result does
not qualify a compiler speedup.

| Inclusive modeled cycles | Independent port | Compiler |
| --- | ---: | ---: |
| Reset | 32,177,859 | 4,505,788 |
| Prefill, including compiler preparation | 5,621,628,052 | 8,882,396,857 |
| 32 decodes, including compiler preparation | 6,255,077,589 | 11,375,721,328 |
| Total | 11,908,883,500 | 20,262,623,973 |

The compiler takes **1.701× as many cycles**. Source/compiler is 0.5877.
Both sides include their session reset and runtime state preparation;
console dumping and immutable checkpoint packing are outside the intervals.
The port specializes the fixed positions and uses persistent cache storage.
The compiler includes its functional stage handoffs. These are complete
implementations with different scheduling choices.

| All 33 complete vocabulary outputs | Independent port | Compiler | Compiler gate |
| --- | ---: | ---: | --- |
| Mean KL against floating reference | 0.0720928786 | 0.0709557105 | Pass |
| Worst-row KL | 0.1721619053 | 0.1858635015 | Fail: exceeds 1.05 × source |
| Matching reference top tokens | 28/33 | 28/33 | Pass |

The source revision is
[`radiance-kernels` `5866983682583486b38ae64a4d1bfa5e7b2a002e`](https://github.com/ucb-bar/radiance-kernels/tree/5866983682583486b38ae64a4d1bfa5e7b2a002e).
Both implementations use the pinned `e4m3_vpu` Rocket-Spike model with
`mem.host_tracking=0`. These are modeled cycles, with no FPGA measurement.
The [session guide](../../independent_session_checks.md) explains the port,
independent smoke replays, numerical rule and reproduction commands.

The archive retains the complete source and compiler console streams, all
1,056,000 logit words per implementation, the floating references, the source's
complete final cache, counters, receipts, source scheduler and original audit
reports. The original complete comparison reopened both measured executables
and checked all 155 encoded checkpoint matrices against each compiler stage.
Large ELF and checkpoint files remain external and are bound by their hashes.
The portable verifier checks the retained observations and recomputes every
numerical metric, the unchanged gates and inclusive timing. It does not reopen
those external executables or repeat the encoded-weight audit.

```sh
python docs/evidence/radiance_rocket_cached_20261023/verify.py
```

Python and NumPy are required. The frozen parser and numerical helpers in
`support.py` are the ones used for the original comparison. A failed numerical
gate is the expected verified result; the verifier exits successfully when the
recorded evidence reproduces it.
