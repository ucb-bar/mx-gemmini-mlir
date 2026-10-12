# SmolVLA attention graph routing (2026-10-11)

The selected MX+VPU profile already supports padded FP8 batched contractions for
the vision and flow attention shapes. The previous precision selector required
an `attn`/`self_attn` module FQN, which the functional model2MLIR captures omit
for these sites. The MX package now selects every source-compatible rank-three
batched matmul for the FP8 catalog; Merlin's structural operand edges identify
the unique `QK -> softmax -> PV` pairs and their distinct operand policies.

`receipt.json` compares the old and new source-bound RV64 catalogs. Prefix
coverage rises from 211 to 243 contractions, and flow coverage rises from 116
to 148. Each stage adds exactly 32 FQN-less FP8 attention contractions,
retains every old binding and format, and has zero unbound selected sites.
Both catalog objects were compiled with Merlin `schedule.plan` and `mk` native
DRAM-loop issuers. Merlin's whole-stage builds linked 243 prefix and 148 flow
source regions into RV64 ELFs with exact catalog coverage checks.
`prefix_pairs.json` and `flow_pairs.json` record the matched
source regions and shapes. These are compiler coverage results; they do not
claim whole-model numerical qualification or VPU fusion.

Reproduce with the pinned profile and the captured MLIR passed to
`tools/audit_merlin_attention_pairs.py --mlir <capture-or-prepared.mlir>
--profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json`.
Build the corresponding source-bound catalog with
`tools/build_merlin_catalog.py --prepared <prepared.mlir> --profile <profile>
--riscv-root <riscv-tools> --out-dir <new-output-directory>` and set
`MERLIN_TARGET_PATH` to this repository. Prefix uses Merlin's MX BF16 prepared
file; flow uses the captured f32 source. Source and object SHA-256 values in
`receipt.json` identify the exact inputs and builds.
