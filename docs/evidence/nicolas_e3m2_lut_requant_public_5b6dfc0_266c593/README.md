# Nicolas E3M2 LUT-index requantization across meshes

Compiler commit `5b6dfc02b0516c5bd5b092686434068bc855ffb4` compiled all
five pinned E3M2 LUT-index requant source programs from fresh
PyTorch/model2MLIR captures. The compiler checked each source recipe and
header, bound the three 6-bit LUT banks at their actual line counts, emitted
data-free public RV64 objects, and linked and ran the results on pinned Spike.

| Shape M×N×K | Mesh | Packed bytes | E8M0 scales |
|---|---:|---:|---:|
| 64×64×64 | DIM8 | 2,048 | 128 |
| 64×64×64 | DIM32 | 2,048 | 128 |
| 128×128×128 | DIM32 | 8,192 | 512 |
| 64×128×128 | DIM32 | 4,096 | 256 |
| 128×64×128 | DIM32 | 4,096 | 256 |

All **20,480 packed bytes** (40,960 4-bit LUT indices) and **1,280 scale
bytes** matched the source goldens. Every case retains bound MLIR, physical
commands, a public object, ELF, Spike log, and hash receipt. This is
numerical source parity on Spike for these five programs; it does not
establish FPGA timing or performance parity.

Reproduce from compiler commit `5b6dfc0`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut e3m2 --symmetric-lut-requant --public-object \
  --mesh-dim "$DIM" --source-shape "$SHAPE" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile "profiles/gemmini-mx-cleanup-266c593/MxDim${DIM}AllGemminiRocketConfig.json" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

Use the five DIM/SHAPE pairs in the table.
