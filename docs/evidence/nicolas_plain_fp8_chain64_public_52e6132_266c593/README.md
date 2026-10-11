# Nicolas's 64³ resident FP8 chain through a public MX object

Compiler commit `52e6132` lowers the pinned source
`matmul_tiled_fp8_64x64_chain.c` (SHA-256 `54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf`)
for RTL revision `266c593f2cb51d7e3fe83fc0317072b585ac3c52` and the
MX-only `MxGemminiRocketConfig` profile. Pinned model2MLIR `e9ded36` captures
two quantized `torch.matmul` sites; the compiler binds the source's packed A1,
B1, B2 and E8M0 scales, places MM1's C1 codes and scales in resident MX
scratchpad, then consumes them in MM2. The public `mx_issue.o` has a ten-buffer
ABI and contains no operands or golden outputs.

The original C program exits successfully on Nicolas's pinned Spike. The
compiler-linked public object also exits successfully, matching all **4,096
C1 FP8 codes, 128 C1 scales, 4,096 C2 FP8 codes, and 128 C2 scales**. Its
issuer C and object are byte-identical to the source-bound reference build.
The receipts are `capture/receipt.json`, `reference/build/artifact_manifest.json`,
`object/object_manifest.json`, and `parity/qualification_manifest.json`.
This qualifies one named connected source program and its numerical result on
Spike. It does not establish FPGA timing or arbitrary C control-flow lowering.

Reproduce with pinned checkouts, a Python environment containing PyTorch,
model2MLIR, MXQuant and compiler dependencies, native `mx-gemmini-opt`, and
Nicolas's RISC-V GCC/Spike. Set `PYTHONPATH` to the MXQuant and model2MLIR
roots. Each command requires a fresh output directory:

```sh
"$PYTHON" -m tools.capture_nicolas_chain \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --matrix-dim 64 \
  --out-dir "$OUT_DIR/capture"
"$PYTHON" -m tools.qualify_nicolas_resident_128 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --connected-frontend-dir "$OUT_DIR/capture" \
  --source-rows 64 --source-width 64 --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR/reference"
"$PYTHON" -m tools.emit_resident_pair_object \
  --mlir "$OUT_DIR/reference/connected_chain.mlir" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --resources-dir "$OUT_DIR/reference/build" \
  --abi-json examples/resident-pair-abi.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR/object"
"$PYTHON" -m tools.qualify_resident_pair_object \
  --object-dir "$OUT_DIR/object" --frontend-dir "$OUT_DIR/capture" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --source-rows 64 --source-width 64 \
  --out-dir "$OUT_DIR/parity"
```
