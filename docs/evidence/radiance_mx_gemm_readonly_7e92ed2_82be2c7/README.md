# Read-only full Radiance MX GEMM roster replay

Compiler commit `7e92ed2` recaptured all **31** Radiance MX GEMM drivers
from PyTorch through model2MLIR `e9ded36`, emitted compiler-issued RV64
programs, and ran them on Nicolas's pinned MX Spike extension from RTL
`266c593`. The source checkout was Radiance `82be2c7`; its 31 driver and
header hashes matched the archived `ee22e0b` baseline despite the newer
repository revision.

All 23 BF16-output and eight requantized-output drivers matched their
complete source goldens. The [reproduction receipt](reproduction.json) records
the source, frontend, simulator, and compiler revisions and both output-index
hashes. The [archive index](index.json) pins all 66 captured receipts and
indices retained here. The header report records zero generated headers.
The source checkout's Git status had the same SHA-256 before and after the
run; the command used `--read-only-source` and wrote its outputs in a separate
task directory. No FPGA was used.

Repeat with a prepared Radiance source checkout and the matching pinned tools:

```sh
python -m tools.reproduce_radiance_roster \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt "$MX_OPT" \
  --radiance-opt "$RADIANCE_OPT" --out-dir "$OUT_DIR" --jobs 2 \
  --compatible-source-revision --read-only-source
```

This is a GEMM source-roster result. It does not assert full-model inference,
all VPU programs, or performance parity.
