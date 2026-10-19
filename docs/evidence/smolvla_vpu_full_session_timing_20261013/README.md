# SmolVLA full-session MX+VPU timing

A linked, checkpoint-backed SmolVLA session completed on the reference implementation’s MX+VPU
functional and timing Spike. The capture uses synthetic inputs and contains
prefix encode, ten flow denoise steps, and action decode. The guest reported
`DONE`, zero memref rank mismatches, and all 45 expected session events. The
runner validated a digest for every declared output and state binding. This
establishes structural execution, not numerical agreement with an independent
reference under the selected MX arithmetic.

| Program | Steps | Modeled cycles | Retired instructions |
| --- | ---: | ---: | ---: |
| Prefix encode | 1 | 201,830,803,146 | 201,051,074,147 |
| Flow denoise | 10 | 411,340,323,807 | 411,129,872,926 |
| Action decode | 1 | 4,236 | 4,232 |
| **Session total** | **12** | **613,171,131,189** | **612,180,951,305** |

The session took 7,889 seconds of host wall time. Flow denoise accounts for
67.08% of the modeled cycle total, averaging 41.13 billion cycles per step;
prefix encode accounts for 32.92%. Action decode has no selected MX/VPU
operation in this profile.

The run used `GEMMINI_MODE=both`, the `e4m3_vpu` timing preset, and
`mem.host_tracking=0`. This override skips CPU-store cache tracking while
retaining MX/VPU command timing. The [guest log](spike.log) records each
`SESSION_PERF` and output digest; the [extension report](gemmini_perf.txt)
records 20,224,685 MX commands, 271,438 VPU commands, and 261,921,984 mesh
busy cycles. Guest instructions are nearly equal to the total modeled cycles,
which makes host-side instruction cost a priority for profiling. Generated
dense adapters currently call `mx_pack_weight_f32` on each invocation; safe
prepacking of immutable operands is a code generation opportunity, subject to
checking each operand's binding and lifetime.

The receipt status is `complete_numeric_qualification_pending` with process
exit zero. Its identities are:

| Artifact | SHA-256 |
| --- | --- |
| Session ELF | `3e8475e55a3bd10f23f68dd937b3d1479d41b1be80e8fc7148ec801c7b98c9e5` |
| Link receipt | `7fecce501fb1b8820c49a885dfa2141fa3340e8b316511166cb90824b974a928` |
| Spike | `de7573f938268f77b8a1729420d3c47aacebed571efc090da73ee87078ce01e9` |
| MX extension | `5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5` |
| Guest log | `398cb33ac9dc3dd6b87c995ab0095dd9cbc4a3787624a57e20494e30cdc0a8fa` |
| Extension report | `4df22c35d475f3e82551275cd57d74fab4fef5e44b0c4a12c2b9611b9c8b3707` |

The [compilation guide](../../compile_e2e_models.md) reproduces the capture,
build, link, and run. Compare future results only with the same inputs,
precision policy, Spike and extension builds, timing preset, and overrides.
These are simulator model cycles, not measured FPGA latency.
