# Radiance MX GEMM source roster on Nicolas's Spike

This archive records compiler-generated RV64 programs executed on the
`gemmini-mx-cleanup` Spike extension at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. The source drivers and data
headers are from Radiance revision `80f84caedbabc663a7433c1da4455b936cca41f3`;
the typed handoffs are from model2MLIR revision
`e9ded36eb85abf2d9097ac4dc11457c825853388`. The compiler revision is
`a541a65aaff2704ca3ac1697c0d8fb1c903a9fc5`.

| Profile | Source cases | BF16 outputs per run | Runs |
| --- | ---: | ---: | ---: |
| `MxDim8AllAsymGemminiRocketConfig` | 23 | 376,832 | 2 |
| `MxDim32AllAsymGemminiRocketConfig` | 23 | 376,832 | 2 |
| `MxE4M3Fp4VpuGemminiRocketConfig` | 2 | 20,480 | 2 |

All 1,548,288 comparisons passed. The first two profiles cover every
BF16-output FP4, FP6, and FP8 MX GEMM driver in the archived model2MLIR
capture. The VPU-enabled profile covers the selected FP8 and FP4 matrix
drivers. Its matrix checks do not exercise VPU commands; separate VPU
qualification is documented in `docs/compiled_mx_pipeline.md`.

The DIM8 and DIM32 reference is derived from Radiance's pinned host model.
The model must first reproduce the checked-in DIM16 source golden exactly.
The target derivation then changes the mesh accumulator schedule and adds
Nicolas's product floor below exponent −16. Its policy and source/target
golden hashes are in each bundle manifest. This is a target-model comparison,
not an assertion that DIM8 or DIM32 output bytes equal the DIM16 header.

Run the two full rosters from the compiler repository root:

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --all-fullout \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim8AllAsymGemminiRocketConfig.json \
  --out-dir /tmp/mx-dim8-fullout
python -m tools.qualify_radiance_mx_base_profile \
  --all-fullout \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim32AllAsymGemminiRocketConfig.json \
  --out-dir /tmp/mx-dim32-fullout
```

The VPU-profile matrix check uses the same command with
`--case fp8 --case fp4` and profile
`MxE4M3Fp4VpuGemminiRocketConfig.json`. The eight quantized-output source
drivers are outside this BF16 roster.

`index.json` binds each profile summary to its 48 archived case receipts.
Each case directory contains the generated command issuer, physical program,
ELF, bound MLIR, golden hashes, and both Spike logs. The two runs have
identical output, ELF, and Spike hashes; their build-log hashes differ because
the output directory is present in some compiler logs. Run
`pytest tests/test_radiance_fullout_mesh_roster_evidence.py` to verify the
archive.
