# Legal Radiance MX GEMM roster on Nicolas's VPU-enabled profile

This archive uses compiler `e9965937463d412442d957826fe019c6f29e5b88`,
Radiance source `80f84caedbabc663a7433c1da4455b936cca41f3`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and Nicolas's
`gemmini-mx-cleanup` RTL/Spike revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`.

The selected `MxE4M3Fp4VpuGemminiRocketConfig` contains the VPU and supports
direct FP8 and FP4 matrix modes. All source MX GEMM drivers using those
precisions were compiled from their archived model2MLIR handoffs and checked
on Nicolas's pinned Spike extension twice:

| Source roster | Drivers | Comparisons per run | Runs |
| --- | ---: | ---: | ---: |
| BF16 output | 18 | 294,912 BF16 values | 2 |
| Quantized output | 6 | 73,728 output bytes and 2,304 E8M0 scales | 2 |

Every output matched. The generated issuers, physical programs, RV64 ELFs,
bundle manifests, source goldens, Spike logs, and repeat receipts are
archived per driver. Only output-directory-dependent build-log hashes differ
between runs.

The source matrix drivers in this roster do not issue VPU commands. Separate
source-derived scalar and chained VPU programs exercise those commands; see
`docs/compiled_mx_pipeline.md`. FP6 requires LUT compute, which this VPU
configuration does not expose. `fp6_rejection.json` records the compiler's
fail-closed profile diagnostic, and the archive test reproduces it.

Run the two legal subsets from the compiler repository root:

```sh
python -m tools.qualify_radiance_mx_base_profile \
  --all-fullout --precision FP4 --precision FP8 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out-dir /tmp/mx-vpu-fullout
python -m tools.qualify_radiance_mx_base_profile \
  --all-requant --precision FP4 --precision FP8 \
  --source-root /path/to/radiance-kernels-80f84ca \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out-dir /tmp/mx-vpu-requant
```

Use fresh output directories for repeat runs. Run
`pytest tests/test_radiance_vpu_legal_roster_evidence.py` to verify the
archive and FP6 rejection.
