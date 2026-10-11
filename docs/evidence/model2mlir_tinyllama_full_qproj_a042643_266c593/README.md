# Complete TinyLlama q_proj on MX Spike

This qualification executes **one complete rank-two projection** from the
reduced random-weight TinyLlama capture: layer-0 `q_proj`, model2MLIR region
`matmul_1`, source shape 8×2048×2048. An actual two-layer TinyLlama forward
supplied the eight input rows and its projection weight. MXQuant encoded the
full 2048-element K axis and all 2048 output columns. The DIM16 target padded
M from 8 to 16. One typed MX contraction lowered through 64 output tiles to
a data-free RV64 RoCC object, then linked to a standalone ELF. Pinned Nicolas
Spike matched the independent hardware arithmetic model on **32,768/32,768
BF16 values**, including all **16,384 model outputs** and 16,384 padded
outputs. The f32 PyTorch result differs from the MX BF16 result by mean
absolute error 0.04136503 and maximum absolute error 0.24947143.

The [summary](summary.json), [operand recipe receipt](input_receipt.json),
[payload manifest](bundle/manifest.json), [physical commands](object/physical_program.json.gz),
[object manifest](object/object_manifest.json), [ELF receipt](run/artifact_manifest.json),
and [Spike log](run/spike.log) bind the source site, all packed bytes, profile,
toolchain, compiler code revision, and execution. The complete quantized
operands and golden are in `bundle/`; the f32 model weights can be regenerated
with the deterministic recipe below. The object embeds zero operand bytes,
zero golden bytes, and zero allocated data bytes. Its 36,998 ordered commands
were compiled at GCC `-O1` to keep large immediate constants in instructions.

Use a Python environment with model2MLIR, MXQuant, xDSL, PyTorch, and
Transformers. `M2M` must have the committed model2MLIR `a042643` source
closure; `MXQ` is MXQuant `b4af543`; `RTL` is Nicolas RTL `266c593` with
its pinned Spike extension. Set `RADIANCE` to the repository containing the
pinned `lib/golden/mx_golden.cpp`. Place outputs in the managed task's `.tmp/`:

```sh
python -m tools.prepare_tinyllama_projection_inputs \
  --model2mlir-root "$M2M" --out-dir "$TASK_DIR/.tmp/tinyllama-inputs"
python -m tools.qualify_model_projection \
  --full-model-mlir docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz \
  --site-region matmul_1 --row-count 8 --column-count 2048 --tile-k 128 \
  --activation-npy "$TASK_DIR/.tmp/tinyllama-inputs/activation.npy" \
  --weight-npy "$TASK_DIR/.tmp/tinyllama-inputs/weight.npy" \
  --model2mlir-root "$M2M" --mxq-root "$MXQ" \
  --reference-root "$RADIANCE" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root "$RTL" --riscv-root "$RISCV" --mx-opt "$MX_OPT" \
  --out-dir "$TASK_DIR/.tmp/tinyllama-qproj-full" --run-spike
```

This result closes the executable lowering path for this one projection.
Whole-model compilation still needs all other projections, attention,
normalization, embeddings, control, KV-cache state, a host execution lane,
and whole-program linking. It does not qualify a complete TinyLlama session
or any pretrained checkpoint.
