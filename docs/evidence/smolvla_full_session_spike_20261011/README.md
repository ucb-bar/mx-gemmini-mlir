# Full-checkpoint SmolVLA session on Nicolas's MX Spike

The [path-free summary](summary.json) and [raw guest console](spike.log) record
one prefix encode, ten recurrent flow steps, and one action decode from a
linked full-checkpoint SmolVLA session. Nicolas's pinned MX Spike extension
exited zero, the guest reached `DONE`, and all 45 expected protocol events
(12 program starts and 33 output/state digests) appeared with zero memref rank
mismatches. The summary binds the run to the linked session receipt and pinned
Spike binaries by SHA-256. No checkpoint bytes or ELF are in this archive.

This is **complete structural execution** of the mixed host/MX session. Its
selected FP4/FP8 contractions use the earlier serial catalog; captured f32
rank-four softmax remains on the host and action decode has no selected MX
contraction. The run has no independent quantized reference for every output
and recurrent state, so it does not establish model numerical correctness or
VPU softmax lowering.
