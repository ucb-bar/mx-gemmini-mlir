# Nicolas FP8 native-loop scale controls

The pinned `matmul_tiled_fp8_128x128_dramloop_nc_2d.c` and
`matmul_tiled_fp8_128x128_dramloop_nc_wait.c` are two N-chunk variants of
Nicolas's 128³ FP8 native DRAM-loop source. Both use two-dimensional scale
uploads. The latter omits the scale-upload fence and sets the hardware wait bit
in `CONFIG_SCALE_MEM` (`rs2[16]`).

For each variant, a fresh PyTorch → model2MLIR capture and source-bound MX
module produced a public data-free RV64 object. The original source executable
and the compiler-linked executable pass independently on pinned Spike. Each
compiler-linked run matches all **16,384 BF16 source-golden outputs**. The
`native_dram_equivalence.json` files audit scale commands, native loop
commands, bank IDs, and output stores. The 2D variant's object is byte-identical
to the previously qualified two-chunk object. These checks establish numerical
and command-operand parity on Spike; they do not establish cycle or FPGA parity.

Reproduce from compiler commit `c2e25ce2c8d5b9701b02837d307b922ae16cd749`,
RTL `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
for variant in nc_2d nc_wait; do
  "$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
    --case "fp8_128x128x128_native_dram_${variant}" \
    --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
    --rtl-root "$MX_RTL_ROOT" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
    --out-dir "$OUT_DIR/$variant"
done
```

The qualifier refuses to overwrite an existing output directory.
