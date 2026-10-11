# Nicolas E4M3 LUT matrices across DIM8 and DIM32

Compiler commit `4f316aca75817acfb31702772ba2f168521b5412` replayed three
more pinned E4M3 LUT source programs through fresh PyTorch/model2MLIR
capture, source payload binding, the public data-free RV64 object compiler,
linking, and pinned Spike. All **24,576 BF16 outputs** matched their source
goldens.

| Source program | Profile | BF16 outputs |
|---|---|---:|
| `matmul_tiled_fp8_e4m3_lut_64x64_dim32.c` | `MxDim32AllGemminiRocketConfig` | 4,096 |
| `matmul_tiled_fp8_e4m3_lut_64x64_nonrequant_dim8.c` | `MxDim8AllGemminiRocketConfig` | 4,096 |
| `matmul_tiled_fp8_e4m3_lut_128x128_nonrequant_dim8.c` | `MxDim8AllGemminiRocketConfig` | 16,384 |

Each case retains frontend and bound MLIR, source recipe, packed operands,
scales, per-row A/B/C LUT banks, physical command program, RV64 object,
generated checker, ELF, Spike log, and hash receipt. These are numerical
source-program replays; the source C performance instrumentation and
requantized sibling programs are outside this result.

Replay from the compiler commit above with RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native
`mx-gemmini-opt`, and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_symmetric_lut_public_suite --jobs 3 \
  --case e4m3_dim32_64 --case e4m3_dim8_64 --case e4m3_dim8_128 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```
