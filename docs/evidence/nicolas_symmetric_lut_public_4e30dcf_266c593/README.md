# Nicolas same-format LUT matrices through public MX objects

Compiler commit `4e30dcfafa7c81dfaec0c49e2d056d14e037ba7a` replayed the three
pinned DIM16 symmetric LUT matrix programs with their dedicated Rocket
profiles: E2M3, E4M3, and E5M2. Fresh PyTorch/model2MLIR captures were
specialized with the checked source headers, lowered to physical MX commands,
compiled by the public `tools.compile_object` CLI, linked into RV64 ELFs, and
run on pinned Spike. All **12,288 BF16 outputs** matched the source goldens.

| Source program | Profile | BF16 outputs |
|---|---|---:|
| `matmul_tiled_fp6_e2m3_lut_64x64.c` | `MxE2M3OnlyGemminiRocketConfig` | 4,096 |
| `matmul_tiled_fp8_e4m3_lut_64x64.c` | `MxE4M3LutGemminiRocketConfig` | 4,096 |
| `matmul_tiled_fp8_e5m2_64x64.c` | `MxE5M2GemminiRocketConfig` | 4,096 |

Each case retains frontend and bound MLIR, source recipe, operand and
per-row LUT payloads, physical command program, data-free object, generated
checker, ELF, Spike log, and hash receipt. This proves numerical output
equivalence for the three named BF16 programs. It does not compile their
performance instrumentation or their requantized sibling programs.

Replay from the compiler commit above with RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native
`mx-gemmini-opt`, and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_symmetric_lut_public_suite --jobs 3 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```
