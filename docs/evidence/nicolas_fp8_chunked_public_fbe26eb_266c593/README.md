# Nicolas FP8 I-chunk source replays

Nicolas's pinned `matmul_tiled_fp8_128x128_chunked.c` and
`matmul_tiled_fp8_128x128_chunked4.c` split the I axis of the same
128×128×128 FP8 matrix multiplication into two or four loop launches. The
compiler captures `torch.matmul` through model2MLIR, binds the checked source
header and selected MX profile, records the source's I-chunk schedule in typed
MLIR, and emits a data-free RV64 object. No handwritten accelerator commands
appear in the linked comparison driver.

For **each** program, the original source executable and the generated object
match all **16,384 BF16 output values** against the checked-in source golden
on Nicolas's pinned Spike. The [two-chunk audit](chunk2/chunk_equivalence.json)
and [four-chunk audit](chunk4/chunk_equivalence.json) independently check the
scale uploads, loop bounds, scratchpad addresses, and `inc_acc_addr` bank
toggle against the pinned source formulas. The compiler issues consecutive
loop launches, with no fence between them. These are numerical and command
schedule results on Spike; they do not establish overlap timing, cycle counts,
or FPGA performance.

Reproduce from compiler commit `fbe26ebd55d558460b9b6c08ddc969c4e607fcd3`,
RTL `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
for chunks in 2 4; do
  "$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
    --case "fp8_128x128x128_chunked${chunks}" \
    --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
    --rtl-root "$MX_RTL_ROOT" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
    --out-dir "$OUT_ROOT/chunk${chunks}"
done
```

The qualifier refuses to overwrite an existing output directory.
