# Nicolas E4M3 LUT-index requantization on DIM8 and DIM32

Compiler commit `45535285f67aa4f5c60e5c05dc810132df74efdf` compiled four
additional named Nicolas E4M3 LUT-index requant programs. Each run began with
a fresh PyTorch/model2MLIR matmul capture, selected the checked-in source
recipe and packed header, emitted a data-free public RV64 object, linked it
with the source data, and compared the complete output on pinned Spike.

| Source program | Profile | Packed bytes | E8M0 scales |
|---|---|---:|---:|
| `matmul_tiled_fp8_e4m3_lut_64x64_requant_dim8` | `MxDim8AllGemminiRocketConfig` | 2,048 | 128 |
| `matmul_tiled_fp8_e4m3_lut_64x64_requant_dim32` | `MxDim32AllGemminiRocketConfig` | 2,048 | 128 |
| `matmul_tiled_fp8_e4m3_lut_128x128_requant_dim8` | `MxDim8AllGemminiRocketConfig` | 8,192 | 512 |
| `matmul_tiled_fp8_e4m3_lut_128x128_requant_dim32` | `MxDim32AllGemminiRocketConfig` | 8,192 | 512 |

All **20,480 packed bytes** and **1,280 scale bytes** matched source goldens.
The source program, profile, generated MLIR, command program, object, ELF,
receipt, and Spike log are in each case directory. This proves numerical
parity for these four named programs on Spike. It does not qualify FPGA
timing, the rectangular DIM32 programs, or the E2M3/E5M2 LUT outputs.

Reproduce from compiler commit `4553528` with RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`. For each table row, run:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut e4m3 --symmetric-lut-requant --public-object \
  --mesh-dim "$DIM" --source-shape "$SHAPE" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile "profiles/gemmini-mx-cleanup-266c593/MxDim${DIM}AllGemminiRocketConfig.json" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```
