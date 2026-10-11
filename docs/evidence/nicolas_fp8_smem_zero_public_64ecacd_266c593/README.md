# Nicolas FP8 64³ zero-base scratchpad readout

The pinned `matmul_tiled_fp8_64x64_smem_mvout.c` computes C into scratchpad
row zero. Its `SPIKE_SIM` source path reads that region through
`gemmini_mx_read_smem`. The compiler generates a public data-free RV64 object
which stores C at the same row and reads it out through RoCC MVOUT commands.
The [command audit](smem_readout_equivalence.json) checks the zero-base compute
destination, 32 readout rows, and their output offsets.

A fresh PyTorch → model2MLIR capture and source-bound typed MX module produce
an object matching **all 4,096 BF16 source-golden outputs** on Nicolas's pinned
Spike. The original C source passes independently. This qualifies numerical
and scratchpad-address parity. The source's CPU scratchpad read and the
compiler's RoCC MVOUT are distinct readout transports; no cycle or FPGA
performance parity is claimed.

Reproduce from compiler commit `64ecacd1fc95a4cfacd894e355ce2e7567997344`,
RTL `266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
  --case fp8_64x64x64_smem_mvout \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

The qualifier refuses to overwrite an existing output directory.
