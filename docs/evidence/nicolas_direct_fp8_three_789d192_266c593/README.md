# Three additional Nicolas FP8 source programs

Compiler commit `789d192aadc3ad6efb65ad4c15bf310000fb47ed` compiled these
three pinned C source programs through fresh PyTorch/model2MLIR capture,
source-payload binding, the public `tools.compile_object` CLI, RV64 linking,
and pinned Spike. All **11,392 numerical outputs** matched their source
goldens:

| Source program | Full comparison |
|---|---:|
| `matmul_tiled_fp8_64x64.c` | 4,096 BF16 values |
| `matmul_tiled_fp8_64x64_requant.c` | 4,096 FP8 codes and 128 E8M0 scales |
| `matmul_tiled_fp8_96x32x32.c` | 3,072 BF16 values |

Each case retains the frontend and bound MLIR, source payload, physical
program, data-free RV64 object, generated checker, ELF, Spike log, and hash
receipt. The source C program supplies operands and golden outputs. The
generated checker compares every numerical output; it does not reproduce the
C test's performance counters or instruction schedule. The earlier
[15-case suite](../nicolas_direct_matrix_suite_9df3384_266c593/README.md)
remains separate, so these results establish 18 distinct direct matrix
programs in total.

Replay from the compiler commit above with RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native
`mx-gemmini-opt`, and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_suite \
  --case fp8_64x64x64 --case fp8_64x64x64_requant --case fp8_96x32x32 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```
