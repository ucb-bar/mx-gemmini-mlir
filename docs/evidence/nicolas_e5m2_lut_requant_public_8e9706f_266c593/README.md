# Nicolas E5M2 LUT-index requantization through a public MX object

Compiler commit `8e9706fe1c9cc6f16d79700c0a5febe41ac06a7d` captured a
64×64×64 PyTorch matmul with model2MLIR, bound Nicolas's checked-in
`matmul_tiled_fp8_e5m2_64x64_requant.c` source header, and lowered a typed
`mx_gemmini.readout_quantized` with `output_format = "fp8_e5m2"` and
`output_projection = "lut"`. The generated data-free public RV64 object was
linked with source data and run on pinned Spike.

All **2,048 packed LUT-index output bytes** and **128 E8M0 scale bytes**
matched `C_proj_hw` and `C_scales_row`. The [receipt](receipt.json),
[bound MLIR](asymmetric_bound.mlir), [object
manifest](object/object_manifest.json), [ELF](physical/asymmetric_program.elf),
and [Spike log](physical/spike.log) retain the full check. This proves
numerical parity for this named E5M2 program on Spike; it does not qualify
other shapes or FPGA timing.

Reproduce from compiler commit `8e9706f` with RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym \
  --symmetric-lut e5m2 --symmetric-lut-requant --public-object \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE5M2GemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```
