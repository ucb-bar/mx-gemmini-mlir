# SmolVLA flow stage: checkpoint weight preparation

Three builds of the same captured `flow_denoise` stage ran to `DONE` on
the pinned MX+VPU Spike model with `GEMMINI_MODE=both`,
`GEMMINI_PERF_CONFIG=e4m3_vpu`, and
`GEMMINI_PERF_SET=mem.host_tracking=0`. The selected profile was
`MxE4M3Fp4VpuGemminiRocketConfig` (`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`)
and the captured stage MLIR SHA-256 was
`2e709838aee113a2678d838dc4389943438e6487793bb3b426b3707868724e17`.

| Weight preparation | Build hash | Modeled cycles | Relative to baseline |
| --- | --- | ---: | ---: |
| Live packing | `951ddc6fc5e0` | 41,113,583,936 | 1.00× |
| First-call cache | `ad715a277f82` | 41,113,584,946 | 1.00× |
| Build-time prepack | `12e9345b6696` | 6,659,754,454 | 6.17× faster |

All three runs have identical hashes for every declared output, zero memref
rank mismatches, 555,735 MX commands, 24,170 loops, 5,444 fences, 479,784
mesh tiles, and 4,252 VPU commands. All 14 distinct schedule and instruction
set hashes match. The prepacked build embeds RHS codes and scales for 116
source-proven checkpoint contractions. The other 32 contractions retain live
packing. The first-call cache pays the packing cost once, so its single-step
result is expected to match the live-packing cycle count.

The [guest logs](prepacked_spike.log) and [extension reports](prepacked_gemmini_perf.txt)
are retained here for every variant: `baseline_*`, `cached_*`, and
`prepacked_*`. The pinned Spike SHA-256 is
`de7573f938268f77b8a1729420d3c47aacebed571efc090da73ee87078ce01e9`;
the extension SHA-256 is
`5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5`.

This is same-input compiler output parity and one-stage timing. It does not
establish independent PyTorch numerical agreement or full-session performance.
