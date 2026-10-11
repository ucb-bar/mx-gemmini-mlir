# Nicolas FP8 two-chunk direct-scale replay

The pinned `matmul_tiled_fp8_128x128_chunked_2d.c` wrapper selects Nicolas's
two-I-chunk 128×128×128 FP8 kernel and loads the A/B E8M0 scale arrays directly
with 2D MX scale commands. A fresh PyTorch → model2MLIR capture, source-bound
typed MX MLIR module, and public RV64 object reproduce **all 16,384 BF16
outputs** on Nicolas's pinned Spike. The original source program independently
passes its full-output check there.

The [command audit](chunk_equivalence.json) checks every 2D scale upload and
both loop launches against the pinned source formulas. The generated object is
byte-identical to the [two-chunk source replay](../nicolas_fp8_chunked_public_fbe26eb_266c593/chunk2/object/mx_issue.o),
which packs identical scale bytes on the host before loading them. This proves
the output and command schedule on Spike, not overlap timing or FPGA speed.

Reproduce from compiler commit `287593c`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_128x128x128_chunked_2d \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
