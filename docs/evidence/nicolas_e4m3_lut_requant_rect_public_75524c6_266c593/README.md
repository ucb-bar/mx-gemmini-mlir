# Rectangular Nicolas E4M3 LUT-index requantization

Compiler commit `75524c6d0666be404ceba247814290e866076c8a` compiled the
two pinned rectangular DIM32 E4M3 LUT-index requant programs from fresh
PyTorch/model2MLIR captures. The compiler bound each source header's A, B,
and output LUT at its actual line count, emitted data-free public RV64
objects, linked them with source data, and ran them on pinned Spike.

| Source program | A/B/output LUT lines | Packed bytes | E8M0 scales |
|---|---|---:|---:|
| `matmul_tiled_fp8_e4m3_lut_64x128x128_requant_dim32` | 32/64/32 | 4,096 | 256 |
| `matmul_tiled_fp8_e4m3_lut_128x64x128_requant_dim32` | 64/32/64 | 4,096 | 256 |

All **8,192 packed bytes** and **512 scales** matched the source goldens.
Each case directory retains the captured and bound MLIR, recipe, resource
manifest, command stream, object, ELF, Spike log, and hash receipt. This is
numerical source parity for these two named programs on Spike. FPGA timing
and other LUT formats remain unqualified.

Reproduce each case from compiler commit `75524c6`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut e4m3 --symmetric-lut-requant --public-object \
  --mesh-dim 32 --source-shape "$SHAPE" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxDim32AllGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```
