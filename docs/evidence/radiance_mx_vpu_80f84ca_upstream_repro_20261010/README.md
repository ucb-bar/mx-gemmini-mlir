# Current Radiance MX+VPU Spike reproduction

On 2026-10-10, compiler `d548fd7`, upstream model2MLIR `e9ded36`, Radiance
`80f84ca`, and Nicolas's MX+VPU RTL/Spike revision `266c593` reproduced two
four-output-tile programs on the pinned Spike extension:

| Case | Source basis | BF16 values checked | Result |
| --- | --- | ---: | --- |
| FP8 GEMM → VPU ×2 | Checked-in Radiance 256×256×256 driver and header | 65,536 | Exact match |
| FP4 GEMM → VPU ×2 | Fixture generated from the checked-in FP8 driver | 65,536 | Exact match |

Each case contains the new index, compiler artifact manifest, and Spike log.
The original archives contain the full frontend, bound MLIR, physical stream,
generated C, and source bundle. All of those hashes, including the ELF and
Spike log, match the originals. Only the compiler provenance fields and linker
log digest changed. The FP4 fixture is derived; it is not a committed Radiance
FP4 256×256 driver. These are Spike functional tests; RTL and FPGA execution
remain separate qualification gates.

Reproduce in a fresh output directory with the two existing commands:

```sh
python -m tools.qualify_radiance_tilewise_vpu_x2 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/fp8-vpu
python -m tools.qualify_radiance_fp4_derived_tilewise_vpu_x2 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --source-root "$RADIANCE_KERNELS_ROOT" --rtl-root "$MX_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/fp4-vpu
```
