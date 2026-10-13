# SmolVLA flow: remove unused fixed-weight host work

The [exact live quantizer build](../smolvla_flow_fast_quant_spike_20261013/README.md)
still transposed and widened fixed checkpoint weights before MX calls, even
though their packed codes and scales were already in the catalog. Its
operation profile attributed 1,563,533,648 cycles to host operations,
1,208,876,680 to MX calls, and 501,696,761 to VPU calls. The instrumented
stage took 3,291,776,646 cycles and attributed 99.46% of them.

The target post-offload rewrite joins the routing sidecar to the catalog by
source region, operation ordinal, and tensor types. For each proven fixed
RHS, it checks the original parameter path, replaces the unused argument
with an empty tensor of the required ABI shape, and erases dead producers.
The flow rewrite changed 116 call operands and removed 424 cast, transpose,
and empty-tensor operations. Dynamic attention operands are unchanged.

| Build | Build hash | Modeled cycles | Relative to live packing |
| --- | --- | ---: | ---: |
| Live packing | `951ddc6fc5e0` | 41,113,583,936 | 1.00× |
| Fixed RHS prepacked, exact lookup | `a0f69bb24b2c` | 3,291,395,119 | 12.49× faster |
| Fixed RHS host work removed | `1e98c45b10b4` | 2,112,653,481 | 19.46× faster |

The last change saves 35.8% of the preceding stage's modeled cycles. All
three declared output hashes, zero memref rank mismatches, 555,735 MX
commands, 24,170 loops, 5,444 fences, 479,784 mesh tiles, and 4,252 VPU
commands match the preceding build. The pruned ELF SHA-256 is
`bda6b07f7fb926069aed68138faeace76981048141fa4ced6866c04d4b392004`.
The [guest log](pruned_spike.log), [extension report](pruned_gemmini_perf.txt),
[fast-build profile report](fast_profile_report.json), and
[profiled guest log](fast_profile_spike.log) retain the raw evidence.

These are same-input, one-stage Spike results under the pinned `e4m3_vpu`
timing preset with `mem.host_tracking=0`. They establish compiler output
parity, not independent quantized PyTorch agreement or full-session timing.
