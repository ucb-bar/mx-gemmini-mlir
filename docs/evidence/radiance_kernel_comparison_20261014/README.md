# Radiance MX kernels as compiler controls

The [receipt](receipt.json) binds the current `spatter-workloads` checkout at
`b6b9bf44` to existing model2MLIR → MX MLIR → Rocket/RoCC → Nicolas Spike
qualifications. The checked driver hashes are unchanged from the archived
Radiance source, so a compatible checkout with those same hashes also works.
The selected profile is
`MxE4M3Fp4VpuGemminiRocketConfig`; FP6 is outside this profile.

| Radiance source kernel | Shape M×N×K | Compiler result on Spike | Weight traffic |
| --- | ---: | --- | ---: |
| FP8 read-once GEMM | 256×64×2048 | 16,384/16,384 BF16 match source golden | 32 weight-tile reads, 128 KiB |
| FP4 read-once down-projection | 256×64×5632 | 16,384/16,384 BF16 match source golden | 88 reads, 176 KiB |
| FP8 four-pass re-stream | 256×64×2048 | 16,384/16,384 BF16 match its source golden | 128 reads, 512 KiB |

The compiler's physical command stream thus expresses the Radiance read-once
weight schedule and its four-pass re-stream control. The equal-shape FP8
read-once and re-stream fixtures use different generated operands, so the
traffic comparison is a schedule property, not an identical-input runtime
comparison. The [read-once archive](../radiance_ws_read_once_1a7bc9c/index.json)
and [re-stream archive](../radiance_restream_1df2c5f/index.json) contain the
checked output and command receipts. The [legal-profile roster](../radiance_mx_vpu_legal_roster_266c593/index.json)
also checks 18 FP8/FP4 BF16 drivers and six requant drivers, twice each.

These are source-shaped controls. The [FP8 payload-bound
MLIR](../radiance_ws_read_once_1a7bc9c/gemm_mxgemmini_ws/payload_bound.mlir)
records `payload_origin = "radiance_source_header_specialization"`: packed
codes and scales are bound from the source header specialization, and the
tile policy is chosen to match the Radiance schedule. The checks establish
lowering and traffic for these shapes and payloads. They do not establish
automatic schedule discovery from an arbitrary model graph.

The source-to-command mapping is visible in those archives. Radiance's FP8
and FP4 read-once drivers set `TILE_M=256`, `TILE_N=64`, `TILE_K=64`; their
compiled `physical_program.json.gz` plans use the same 256×64×64 tile and
read each K-wave's weight tile once. The re-stream driver sets `TILE_M=64`
and loops over four M blocks; its compiled plan uses four 64×64×64 output
tiles and reads the same 128 KiB weight matrix four times. These are checked
tile and traffic properties, not evidence that the Muon launch, overlap, or
total runtime matches Radiance.
The archived physical programs have 576 weight-move steps for FP8 read-once,
880 for FP4 read-once, and 2,304 for FP8 re-stream. The last count is
exactly four times the equal-shape FP8 read-once count. A step is a physical
program entry; it is not a complete kernel invocation or a cycle.

Other Radiance MX kernel families have narrower compiler controls:

| Family | Existing compiler evidence | Remaining whole-kernel gap |
| --- | --- | --- |
| Batched GEMV projections | [Four FP8/FP4 source shapes](../radiance_batched_gemv_552a16b/index.json), all BF16 outputs and one-read weight traffic checked on Spike | Source launch and Muon integration are separate. |
| TinyLlama GQA flash attention | [All 16 source P/V cutpoints](../radiance_gqa_pv_roster_80f84ca/index.json) pass the MX output checks on Spike; the [final recurrence](../radiance_gqa_final_recurrence_80f84ca/index.json) reconstructs source output on the corrected functional model | Compiler still needs Muon softmax, online accumulation, finalization, and mixed-engine handoff. |
| Full `model_chain` programs | Captured TinyLlama and SmolVLA graphs have MX+VPU compiler sessions | Radiance's existing connected programs use scalar Muon stages; there is no independently stitched MX+VPU full-model source run. |

The GQA cutpoint control uses the archived corrected functional model and
source-produced P tiles. The source generator's `O_gold` differs from the
executed source output at 31,852 of 32,768 BF16 positions; the 65,536-value
PV check is therefore **not** a whole-attention correctness claim.

These are **compiler-on-Spike output and command comparisons**. The original
Radiance drivers use Muon to issue MX work on the combined SoC; no paired
Radiance-source versus compiler cycle count exists for these drivers on
Nicolas's standalone Rocket Spike model. The published Radiance flash-attention
cycles also include Muon SIMT softmax and overlap, which Nicolas's MX+VPU Spike
does not run. They cannot be placed in a speedup ratio with the current model
cycles. See the [GQA source qualification](../../compiled_mx_pipeline.md#gqa-qk-numerical-domain-and-source-derived-candidate)
for the separate MX QK/PV numerical controls.

In particular, [Radiance's 64×256×128 flash-attention
kernel](https://github.com/ucb-bar/radiance-kernels/tree/main/kernels/flash_attention_mx)
reports 42,364 cycles per tile for a steady-state mixed Muon+MX RTL schedule.
The slide's 43,523 compiler cycles are for one dense MX+VPU attention object
on Rocket Spike. Equal dimensions do not make those timers, engines, or
simulators equivalent. The Radiance source also documents schedule-sensitive
correctness for that peak configuration.

The Radiance `model_chain` full-model programs currently use scalar Muon code;
their model or FPGA cycles are likewise not an MX+VPU compiler baseline. The
full TinyLlama and SmolVLA slide rows instead use Nicolas's MX+VPU Spike, and
their issuer-swap qualifier still applies.

Regenerate this receipt against a prepared Radiance checkout with:

```sh
python3 -B docs/evidence/radiance_kernel_comparison_20261014/build_receipt.py \
  --radiance-root /path/to/radiance-kernels \
  --out docs/evidence/radiance_kernel_comparison_20261014/receipt.json
```
