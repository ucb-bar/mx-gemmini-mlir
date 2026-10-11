# Current model2MLIR small LLaMA precision probe

The [frontend graph](small_llama.model2mlir.mlir.gz) is a complete two-layer
**small random-weight LLaMA-style model** from model2MLIR `e9ded36`'s
`workloads/small_llama/loader.py`, captured through the FX importer with
zero opaque operations. It is a compiler probe, not a TinyLlama 1.1B
checkpoint or a source-parity run. The [receipt](receipt.json) pins the
model2MLIR revision, loader and graph digests, and MX+VPU profile.

The [worklist](worklist.json) identifies 19 contractions: 15 rank-two
linear matmuls and four batched attention matmuls. The [precision
proposal](precision_proposal.json) assigns FP4 to nine aligned linear
matmuls and FP8 to both QK/PV pairs under `blocks.0.attn` and
`blocks.1.attn`. Six MLP matmuls remain on the host in this proposal
because dimension 344 needs MX padding. The plan records the required
32×352×128 or 32×128×352 padded shapes and FP4 as the candidate after
padding. Each 8×8 attention score also needs a 32×32 hardware tile; the
proposal records those padded QK/PV shapes.

Two of the six padding candidates are now compiled and executed as separate
FP4 programs: see the [layer-0 gate/down archive](../model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593/README.md).

This is **not an executable whole-model MX program**. It makes the next
lowering steps explicit: quantize and pad the six MLP sites, lower batched
FP8 attention with masked/cropped lanes, then connect all operators and
model state to an executable host/MX program. No FPGA was used.

Regenerate with the pinned model2MLIR checkout, its Python environment,
and the MX RTL checkout:

```sh
python -m tools.capture_small_llama_precision_probe \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --out-dir "$OUT_DIR"
```
