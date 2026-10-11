# Nicolas E4M3 LUT-index requantization through a public MX object

Compiler commit `74bceebfb16b647425de0d322fcfa60118d0d817` captured a
64×64×64 PyTorch matmul with model2MLIR, specialized it using Nicolas's
checked-in `matmul_tiled_fp8_e4m3_lut_64x64_requant.c` and data header, and
lowered the typed `mx_gemmini.readout_quantized` with `output_projection =
"lut"` to physical Rocket/RoCC commands. The public object compiler emitted
a data-free RV64 object, which was linked with the checked source payload and
run on pinned Spike.

The generated executable matched all **2,048 packed output bytes** (4,096
4-bit LUT indices) and **128 E8M0 scale bytes** against `C_proj_hw` and
`C_scales_row`. The object ABI exposes a 2,048-byte packed-index write buffer
and a 128-byte scale write buffer. This is numerical parity for this named
source program on Spike; it does not establish FPGA timing parity or complete
coverage of Nicolas's other requantized LUT programs.

The [receipt](receipt.json), [bound MLIR](asymmetric_bound.mlir),
[object manifest](object/object_manifest.json), [physical command
program](object/physical_program.json), [ELF](physical/asymmetric_program.elf),
and [Spike log](physical/spike.log) retain the inputs, hashes, output
comparison, and generated artifacts.

Reproduce from compiler commit `74bceeb` with RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut e4m3 --symmetric-lut-requant --public-object \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3LutGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```
