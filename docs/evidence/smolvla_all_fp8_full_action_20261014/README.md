# Full all-FP8 SmolVLA action control

This control runs the captured SmolVLA session on Nicolas's MX+VPU Spike:
one prefix encode, ten flow denoising steps, and one action decode. All 243
prefix and 148 flow matrix sites bind to direct E4M3 × E4M3. The selected
`MxE4M3Fp4VpuGemminiRocketConfig` has no operand LUT. The VPU and remaining
host operations follow the compiled session route.

The archived link and catalogs bind this run to the same captured programs,
checkpoint weight blob, target profile, arena layout, and output ABI as the
[corrected mixed FP4/FP8 control](../smolvla_scalar_full_action_20261014/README.md).
Only the prefix and flow stage objects change. The ten flow outputs and final
action are compared with the captured unquantized FP32 action trajectory.
Changing precision changes the output values, so any difference in total
cycles is a precision-policy cost comparison, **not** an equal-output
compiler-versus-source speedup. Neither comparison establishes robot task
success or agreement with an independently quantized MX reference.

| Captured 12-step session | All FP8 | Corrected mixed FP4/FP8 |
| --- | ---: | ---: |
| Final action cosine to FP32 | **0.988524** | 0.058102 |
| Final normalized RMSE to FP32 | **0.246502** | 1.830469 |
| Prefix modeled cycles | 72,012,507,511 | 201,830,803,145 |
| Ten flow steps, modeled cycles | 108,207,126,169 | 114,684,795,994 |
| Complete session, modeled cycles | **180,219,637,915** | 316,515,603,374 |

All ten scalar timestep outputs follow the FP32 recurrence exactly. The
final action is byte-identical to the last flow state. Flow-step cosine to
FP32 starts at 0.999964 and ends at 0.988524; the mixed control starts at
0.997092 and ends at 0.058102. All-FP8 takes 43.1% fewer modeled cycles
over these completed session programs, with different output values. The
timing setting is `e4m3_vpu` with `mem.host_tracking=0` on Nicolas's pinned
Spike extension. These counts are modeled Spike cycles, not FPGA latency.

The [full-model compilation guide](../../compile_e2e_models.md) describes
capture, precision overrides, linking, and Spike execution. The archived
`prefix_encode_overrides.json` and `flow_denoise_overrides.json` name the
promoted sites; the stage catalogs record every selected matrix binding.
The run receipt records the executable, Spike extension, timing setting,
per-step counters, and output digests.

Verify the portable archive from the repository root:

```sh
python -B docs/evidence/smolvla_all_fp8_full_action_20261014/archive.py --verify
```

The verifier checks the archived hashes, profile and weight identity against
the mixed control, all-FP8 binding coverage, each 6,400-byte action output,
the exact scalar timestep recurrence, and the stored FP32 metrics.
