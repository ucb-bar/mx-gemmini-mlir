# Connected FP8 MX/VPU width expansion

Compiler commit `a1c0b4d` was cloned from the published
`handwritten-implementation` branch and run under two RTL-derived profiles:

| Profile | MM1 | MM2 | Spike comparisons |
| --- | --- | --- | --- |
| `MxE4M3Fp4VpuGemminiRocketConfig` | 64×64×64 | 64×96×64 | 4,096 C1 BF16, 4,096 C1 FP8, 128 C1 scales, 6,144 C2 FP8, 192 C2 scales |
| `MxE4M3VpuGemminiRocketConfig` | 64×64×64 | 64×128×64 | 4,096 C1 BF16, 4,096 C1 FP8, 128 C1 scales, 8,192 C2 FP8, 256 C2 scales |

Both rows had zero mismatches and Spike exit code zero. The archived
model2MLIR capture, connected typed MLIR, physical commands, data-free RV64
RoCC object, standalone ELF, and Spike log are bound by
[`archive_manifest.json`](archive_manifest.json). The static test
`tests/test_wide_vpu_pair_evidence.py` checks every archived file digest,
source input lineage, output counts, object ABI, and typed graph re-lowering.

**Numerical scope:** Nicolas's original 64×64 MM1, VPU, and first 64 MM2
columns come from RTL revision `266c593`. Each added B2 column is a
sign-flipped copy of an original source column. The wider C2 expectations
come from the pinned `fp8_matmul_model.py` whose digest is recorded in the
case receipts. These are source-derived compiler cases, not unchanged
Nicolas source kernels, RTL timing results, or qualification of FP4/FP6 VPU
matrix modes.

To reproduce either row from a checkout of the compiler commit, set
`MODEL2MLIR_ROOT` to model2MLIR `e9ded36`, `MXQUANT_ROOT` to MXQuant
`b4af543`, `MX_RTL_ROOT` to Gemmini `266c593` with its software submodules,
and `RISCV_ROOT` to the pinned RV64 GCC/Spike toolchain. Run from the compiler
repo with a Python environment containing Torch, xDSL, and model2MLIR's
dependencies:

```sh
PYTHONPATH=".:$MXQUANT_ROOT" python -m tools.replay_wide_vpu_pair \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --profile MxE4M3Fp4VpuGemminiRocketConfig \
  --second-width 96 --out-dir /new/mx-vpu-n96
```

For the second row, select `MxE4M3VpuGemminiRocketConfig`, width `128`,
and a distinct output directory. `mx-gemmini-opt` must be built from the
same compiler checkout. The replay refuses to overwrite its output directory
and checks the selected source revisions before capture.
