# Nicolas E2M3 and E5M2 LUT-index requantization across meshes

Compiler commit `b94c3afb1635094e854794d434b6f4ac35b8d953` compiled ten
more checked-in Nicolas source programs from fresh PyTorch/model2MLIR captures.
The source recipe selects E2M3 or E5M2 from the named C program and header,
checks the profile's legal compute tuple, binds each LUT bank at its source
size, and lowers a typed packed-index readout. The public compiler emits a
data-free RV64 object for each case. Every generated executable ran on pinned
Spike and matched all packed code and E8M0 scale bytes in the source header.

Each format covers these five source shapes and meshes:

| Shape M×N×K | Mesh | Packed bytes | E8M0 scales |
|---|---:|---:|---:|
| 64×64×64 | DIM8 | 2,048 | 128 |
| 64×64×64 | DIM32 | 2,048 | 128 |
| 128×128×128 | DIM32 | 8,192 | 512 |
| 64×128×128 | DIM32 | 4,096 | 256 |
| 128×64×128 | DIM32 | 4,096 | 256 |

Across both formats, **40,960 packed bytes** (81,920 4-bit LUT indices) and
**2,560 scale bytes** matched. The case directories retain captured and
bound MLIR, recipes, source data, command streams, objects, ELFs, Spike logs,
and hash receipts. These results establish numerical parity on Spike for the
ten named programs. They do not establish FPGA timing or performance parity.

Reproduce each case from compiler commit `b94c3af`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut "$FORMAT" --symmetric-lut-requant --public-object \
  --mesh-dim "$DIM" --source-shape "$SHAPE" \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile "profiles/gemmini-mx-cleanup-266c593/MxDim${DIM}AllGemminiRocketConfig.json" \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

Use `FORMAT=e2m3` or `e5m2` and the five DIM/SHAPE pairs in the table.
