# 128-cubed connected MX/VPU source-derived graph

Published compiler commit `cc4859c` was cloned from
`handwritten-implementation`, then used to capture Nicolas's two 128³ FP8
matmuls with model2MLIR `e9ded36`. The compiler bound the original source
operand bytes, lowered one typed MM1 → BF16 readout → VPU ×2 → resident
requant → MM2 graph, emitted a data-free RV64 RoCC object, linked an ELF, and
ran it on Nicolas's pinned Gemmini Spike extension.

| Target profile | C1 BF16 | C1 FP8 | C1 E8M0 scales | C2 FP8 | C2 E8M0 scales |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MxE4M3Fp4VpuGemminiRocketConfig` | 16,384 | 16,384 | 512 | 16,384 | 512 |
| `MxE4M3VpuGemminiRocketConfig` | 16,384 | 16,384 | 512 | 16,384 | 512 |

Every comparison had zero mismatches and both Spike runs exited zero. The
profiles have different hashes; the FP8 object, ELF, and Spike log hashes
match. [`archive_manifest.json`](archive_manifest.json) records all 62
archived file digests, including the frontend capture, typed graph, physical
stream, runtime resources, object, ELF, and full comparison log. The CI test
`tests/test_square_128_vpu_pair_evidence.py` checks those digests, compares
the original source input hashes with the existing plain 128³ source archive,
checks the BF16 ×2 reference, re-lowers the typed graph, and rejects changed
input bytes or a scratchpad output overlapping the live BF16 tile.

**Numerical scope:** Nicolas's 128³ source provides the unchanged A1/B1/B2
codes and scales and MM1 BF16 golden. The source does not contain this VPU
chain. The VPU ×2 C1 reference is derived from those BF16 bits; the C2
reference comes from Nicolas's pinned FP8 mesh model, after that same model
is checked against the source's original C2 golden. This is a functional
Spike result, not an RTL timing or FPGA qualification.

To reproduce from a checkout of compiler `cc4859c`, set `MODEL2MLIR_ROOT`
to revision `e9ded36`, `MXQUANT_ROOT` to `b4af543`, `MX_RTL_ROOT` to Gemmini
`266c593` with its software submodules, and `RISCV_ROOT` to the pinned RV64
GCC/Spike toolchain. Use a Python environment with Torch and model2MLIR's
dependencies and build `mx-gemmini-opt` in this checkout:

```sh
PYTHONPATH=".:$MXQUANT_ROOT" python -m tools.replay_wide_vpu_pair \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --profile MxE4M3Fp4VpuGemminiRocketConfig \
  --first-width 128 --second-width 128 --out-dir /new/mx-vpu-128
```

Repeat with `MxE4M3VpuGemminiRocketConfig` and a distinct output directory
for the second profile. The replay checks source revisions before capture
and refuses to overwrite its output directory.
