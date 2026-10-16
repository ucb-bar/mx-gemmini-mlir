# Capture-bound execution policy: two-layer TinyLlama

The compiler built the first two pretrained TinyLlama layers with 20 MX
contractions and 65 selected VPU source sites. The policy kept softmax and the
two attention-score adds on the host. Its linked ELF completed on Nicolas's
MX+VPU Spike extension with `rv64gc_zicntr`, `GEMMINI_MODE=both`, the
`e4m3_vpu` timing preset, and `mem.host_tracking=0`.

The final 32,000-element logit row has SHA-256
`c4322c6551c769381a794710a2395f9d261ba9d5312e7df50436c6ef1e1878ad`.
This exactly matches the previously archived compiler
[65-site class control](../tinyllama_l2_vpu_class_controls_20261014/README.md).
The guest reported `DONE`, zero memref rank mismatches, and **577,315,560**
cycles. The earlier control used a different timing setup, so its cycle count
is not a matched performance comparison. Identical output gives this policy
the control's 0.9915 cosine to captured PyTorch logits and top token 5056 for
this one input. It does not qualify the full 22-layer model or a cached
session.

The [receipt](receipt.json) pins the capture, profile, policy, extension,
output digests, and timing configuration. The [Spike log](spike.log) preserves
the guest and accelerator counters. The stage was first run with `rv64gc`,
which trapped on `rdcycle`; the archived run reused the same ELF with
`rv64gc_zicntr`. The stage builder now defaults to that ISA.
