# Four checkpoint-backed linked sessions

The [summary](summary.json) records complete RV64 ELF **links** for
TinyLlama, DeepSeek-R1-Distill-Qwen-1.5B, Gemma 2 2B, and SmolVLA. Each
combines the receipt-bound stage ELFs in the
[nine-stage archive](../four_checkpoint_stage_links_20261011/README.md)
with Merlin's multi-program state scheduler, one byte-verified checkpoint
weight image, and the MX catalog objects. No checkpoint bytes or ELFs are
stored in this archive.

TinyLlama, Qwen, and Gemma link one prefill followed by 32 decode steps.
SmolVLA links one prefix encode, ten recurrent flow steps, and host-only
action decode. Every invocation resets temporary guest allocations while
keeping Merlin's persistent output and state buffers. The summary records
the exact stage source digests, selected MX call counts, result-record counts,
ELF SHA256 and size, shared-weight SHA256 and size, and the disjoint Spike
arena layout. All four use Nicolas's pinned
`MxE4M3Fp4VpuGemminiRocketConfig` profile.

This is **structural link evidence**. The ELFs were submitted to the pinned
MX Spike extension, but a link receipt is not a guest `DONE`, a complete
output/state trace, quantized numerical parity, or a VPU placement result.
The current catalog routes selected FP4 and FP8 contractions; captured f32
rank-three softmax remains on Merlin's compiled host lane pending an explicit
BF16 VPU legalization and numerical policy. Do not infer configured FPGA
memory fit from the unbounded Spike memory maps in this archive.
