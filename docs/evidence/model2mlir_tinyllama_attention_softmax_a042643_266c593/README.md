# TinyLlama model-derived scores through MX VPU softmax

This archive tests the first decoder attention head of the reduced,
randomly initialized TinyLlama capture. The current pinned model2MLIR
`a042643` export identifies QK region `matmul_4`, softmax region
`softmax_0`, and PV region `matmul_5`. A seeded execution of that export
confirmed that the softmax input is QK multiplied by **1/8**, with the
28 positions above the 8×8 causal diagonal masked. The input receipt records
the exact full-model graph, QK operands, floating softmax input, and floating
probabilities.

The separate [compiled QK head](../model2mlir_four_model_fp8_qk_vpu_a042643_266c593/README.md)
supplies 16×32 BF16 output bytes from pinned MX Spike. The host applies the
verified 1/8 scale, an infinite causal mask, and padding to those bytes. A
model2MLIR capture of a BF16 16×32 softmax lowers to the existing six VPU
operations in typed MX MLIR, then to a **data-free, 17-command RV64 object**.
The reference implementation’s pinned Spike model matched all **512 BF16 outputs** against its
independent `vpu_ref_exec` arithmetic reference. The eight real rows have
zero output at every masked position. The real 8×8 probability block has
mean absolute error **0.00279** versus the captured PyTorch f32 result; this
is a diagnostic for the reduced random-weight model, not a quality target.

Two clean runs produced identical model2MLIR capture, bound MLIR, physical
program, object, ELF, Spike log, and other recorded artifacts. See
`reproduction.json` for 15 matching SHA-256 values. The archive also contains
the source floating tensors, the packed BF16 score, the 8×8 BF16 result,
generated issuer C, and the source-derived checker driver.

This is an **artifact-bound seam**: the QK output was read from a separate
Spike run and scaled and masked on the host before the VPU program ran. The
compiler does not yet issue an on-device QK→scale/mask→softmax sequence or
feed this VPU result into the separately compiled PV head. That buffer routing
and mixed-engine lowering remain necessary for full attention and full-model
inference.

Check the archive with:

```sh
python -m pytest -q tests/test_tinyllama_attention_softmax_evidence.py
```

To regenerate, first use `tools.prepare_tinyllama_attention_head_inputs`
with `--stage qk` to reproduce the QK float arrays. Run
`tools.prepare_tinyllama_attention_softmax_inputs` with those arrays, the
archived full TinyLlama MLIR graph, and pinned model2MLIR source. Then run
`tools.qualify_tinyllama_attention_softmax` with that capture, the archived
QK bundle, `MxE4M3Fp4VpuGemminiRocketConfig.json`, pinned RTL, RISC-V tools,
and a native `mx-gemmini-opt`. Both tools refuse an existing output directory.
