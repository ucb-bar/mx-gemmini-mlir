# Radiance MX quantized outputs on Nicolas's Spike

This archive records all eight quantized-output MX GEMM drivers from the
pinned Radiance source and model2MLIR capture. The compiler revision is
`13417df20c73480a51736af38a56141438d9a3a5`; Radiance is
`80f84caedbabc663a7433c1da4455b936cca41f3`, Nicolas's MX RTL and Spike
extension are `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, and
model2MLIR is `e9ded36eb85abf2d9097ac4dc11457c825853388`.

| Rocket profile | Drivers | Output bytes per run | E8M0 scales per run | Runs |
| --- | ---: | ---: | ---: | ---: |
| `MxDim8AllAsymGemminiRocketConfig` | 8 | 90,112 | 3,328 | 2 |
| `MxGemminiRocketConfig` (DIM16) | 8 | 90,112 | 3,328 | 2 |
| `MxDim32AllAsymGemminiRocketConfig` | 8 | 90,112 | 3,328 | 2 |

Every comparison passed on Nicolas's pinned Spike extension. The DIM16 run
uses the exact checked-in source BF16, code, and scale goldens. DIM8 and
DIM32 first require Radiance's pinned host model to reproduce the DIM16 BF16
golden, then derive target-mesh BF16 with Nicolas's product floor. The
Radiance output quantization is recomputed from that target BF16. The
archive keeps the original source codes and scales beside the target values;
all eight cases change at least one code on each non-DIM16 mesh.

The emitted executable runs an MX contraction, reads BF16, and executes the
typed Radiance header requantization as a host epilogue. These comparisons
qualify that source convention. They do not claim numerical parity for the
MX hardware requantizer, whose format and rounding rules differ.

Reproduce a target roster from the compiler repository root:

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --all-requant \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim8AllAsymGemminiRocketConfig.json \
  --out-dir /tmp/mx-dim8-requant
```

Choose `MxGemminiRocketConfig.json` or
`MxDim32AllAsymGemminiRocketConfig.json` for the other two profiles. Repeat
with a new output directory to check artifact stability. `index.json` binds
both qualification runs to each case's generated issuer, physical program,
ELF, source and target codes/scales, model provenance, and Spike logs. Run
`pytest tests/test_radiance_requant_mesh_roster_evidence.py` to verify the
archive.
