# Sixteen additional Nicolas FP8 and FP4 matrix programs

Compiler commit `28936e2f51e0923ad9553b1c028389372a06ae39` replayed **16 pinned
source programs** with fresh PyTorch/model2MLIR capture, source-payload
binding, the public `tools.compile_object` CLI, RV64 linking, and pinned
Spike. All **126,432 numerical outputs** matched their source goldens.
Seven cases use the DIM16 profile, four use DIM32, and five use DIM8. The
cases include rectangular FP8 matrices, FP4 matrices, BF16 readout, and
FP8/FP4 requantized readout. `index.json` records every source driver and
header hash, selected profile, object and ELF hash, Spike log hash, and exact
comparison extent.

Each case retains the frontend and bound MLIR, pinned source payload,
physical command program, data-free RV64 object, generated checker, ELF,
Spike log, and receipt. The source C program supplies operands and golden
outputs. The checker compares every numerical output; it does not reproduce
the C test's performance counters or instruction schedule. Together with the
[earlier 18 direct cases](../nicolas_direct_fp8_three_789d192_266c593/README.md),
this establishes **34 distinct direct matrix programs**.

Replay from the compiler commit above with RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native
`mx-gemmini-opt`, and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_suite --jobs 4 \
  --case fp8_64x96x64 --case fp8_64x96x64_requant \
  --case fp8_96x32x32_requant --case fp8_96x96x64_requant \
  --case fp8_32x32x32_requant --case fp4_128x128x128 \
  --case fp4_128x128x128_requant \
  --case fp4_64x64x64_requant_dim32 \
  --case fp4_64x64x64_nonrequant_dim32 \
  --case fp4_128x128x128_nonrequant_dim32 \
  --case fp4_128x128x64_requant_dim32 \
  --case fp8_64x64x64_single_dim8 \
  --case fp8_128x128x128_single_dim8 \
  --case fp8_64x64x64_requant_dim8 \
  --case fp8_128x128x128_requant_dim8 \
  --case fp4_64x64x64_requant_dim8 \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```
