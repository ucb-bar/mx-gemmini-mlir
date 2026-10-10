# Fifteen direct Nicolas matrix cases through the public MX object compiler

Compiler commit `9df3384937e89b4b79f5234d68379fdf5a5388ef` ran the complete
registered direct-matrix suite from fresh PyTorch/model2MLIR captures,
source-payload binding, `tools.compile_object`, RV64 linking, and pinned Spike.
The index records **15 passed cases** and **163,072 compared items**. The sum
combines BF16 values, FP8 codes, packed FP4/FP6 bytes, and E8M0 scales.

This revision adds four checked Nicolas source programs:

| Source case | Comparison |
|---|---:|
| `matmul_tiled_fp8_32x32x32.c` | 1,024 BF16 values |
| `matmul_tiled_fp8_64x256x64_requant_dim32.c` | 16,384 FP8 codes and 512 scales |
| `matmul_tiled_fp8_128x128x256_requant_dim32.c` | 16,384 FP8 codes and 512 scales |
| `matmul_tiled_fp4_128x128x512.c` | 16,384 BF16 values |

For those four cases, this directory contains the frontend MLIR, bound MLIR,
checked source payload, physical commands, data-free object, driver, ELF,
Spike log, and receipt. The other eleven cases have receipts here and complete
artifacts in their earlier per-case archives. The source C program is used to
pin the packed arrays, shape, and golden result. The generated driver checks
every numerical output; this is not a compilation of the C test's performance
instrumentation or arbitrary control flow.

Replay from this compiler commit with the RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native `mx-gemmini-opt`,
and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_suite \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```

The suite refuses a changed source, header, RTL revision, frontend revision,
profile, or output extent. `index.json` records the hashes of each compiled
object, ELF, Spike log, and receipt.
