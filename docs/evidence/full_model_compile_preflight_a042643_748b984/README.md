# Four-model whole-program compile preflight

The [index](index.json) and compressed MLIR captures record an attempted
model2MLIR→MX handoff against `MxGemminiRocketConfig`. Source revisions and
exact graph hashes are in the index. Each capture has **zero opaque frontend
ops**. None compiles to an MX object or target executable. The recorded
`ParseError` is the first refusal when the current `tools.compile_object`
classifier is given raw model2MLIR linalg syntax. Even a general parser would
still need whole-graph lowering, weight/state binding, CPU/MX partitioning,
linking, and numerical checks.

The captures deliberately use reduced random-weight structures:

- TinyLlama: 2/22 decoder layers, sequence length 8, tokens to logits.
- DeepSeek-R1-Distill-Qwen-1.5B: 1/28 layers, sequence length 4, tokens to logits.
- Gemma 2 2B: 2/26 layers, sequence length 4, installed `Gemma2Config` defaults;
  the gated model config was not in the offline cache.
- SmolVLA: 1 VLM and 1 expert layer, random weights, 64×64 image, one denoise
  step. The full prefix→flow→action session was not captured here.

The captures used the clean committed model2MLIR `a042643` source closure.
TinyLlama and Qwen used their `workloads/*/loader.py` with `M2M_*_LAYERS`
set as above and session mode cleared, then `m2m convert --backend
fx_importer`. Gemma used its loader with `AutoConfig.from_pretrained`
returning the installed default `Gemma2Config()` for the offline smoke path.
SmolVLA used the same loader with LeRobot 0.5.1, Transformers 5.3.0,
`M2M_SMOLVLA_PRETRAINED=0`, `M2M_SMOLVLA_SESSION=` and the layer/image
overrides above. The files are graph evidence; no checkpoint weights are
archived.

Check the archive and compiler refusal with:

```sh
python -m pytest -q tests/test_full_model_compile_preflight_evidence.py
```

The [acceptance criteria and integration steps](../../full_model_compilation.md)
keep the distinction between frontend capture and full-model target execution
explicit.
