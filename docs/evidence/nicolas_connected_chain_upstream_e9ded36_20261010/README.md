# Current model2MLIR connected MX/VPU chain

This run recaptured Nicolas's 64³ two-matmul graph with upstream model2MLIR
`e9ded36` and MXQuant `b4af543`, then compiled one ordered program for
MM1 → VPU ×2 → resident requant → MM2 against Nicolas RTL and Spike
`266c593`. The source bound operands and expected outputs remain Nicolas's
checked-in chain header. The capture has two quantized MX sites and no opaque
calls. The Spike program matched 4,096 C1 BF16 values, 8,192 C1/C2 FP8 codes,
and 256 C1/C2 E8M0 scales.

The source MLIR is compressed as `source.mlir.gz`. `capture_receipt.json`
records its uncompressed SHA-256 and the hashes of the handoff, profile-bound
MLIR, and quantization manifest. `artifact_manifest.json` ties the connected
MLIR to the physical stream, object files, ELF, extension, and Spike log.
The executable, objects, extension, and Spike log are byte-identical to the
earlier `7485a829` frontend run. The frontend and connected MLIR digests
changed with the upstream frontend version. This remains the source-specific
64³ chain; arbitrary multi-site graph scheduling is still open.

Reproduce with the existing tools and a clean output directory for each step:

```sh
python -m tools.capture_nicolas_chain \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /new/chain-capture
python -m tools.qualify_nicolas_vector_requant \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --with-resident-matmul \
  --frontend-bound-mlir /new/chain-capture/nicolas_chain.profile_bound.mlir \
  --frontend-receipt /new/chain-capture/receipt.json \
  --connected-ssa --out-dir /new/chain-spike
```
