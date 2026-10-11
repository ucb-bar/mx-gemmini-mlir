# TinyLlama attention head across three MX programs

This archive follows one head in the first decoder attention layer of the
reduced, randomly initialized TinyLlama capture from model2MLIR `a042643`.
The full model MLIR names QK `matmul_4`, softmax `softmax_0`, and PV
`matmul_5`. The [QK archive](../model2mlir_four_model_fp8_qk_vpu_a042643_266c593/README.md)
provides the compiler-generated FP8 MX output from pinned Spike. The
[softmax archive](../model2mlir_tinyllama_attention_softmax_a042643_266c593/README.md)
applies the source 1/8 scale and upper-causal mask on the host, then runs a
typed VPU softmax object on that QK output. This archive decodes the **exact
eight-by-eight BF16 probabilities from the VPU Spike log** and supplies them
as the PV operand with the original captured V tensor.

The PV compiler uses the same FP8 E4M3 and E8M0 scale lowering as the
isolated PV head. Its RV64 issuer is data-free and has the **same object
SHA-256** as that isolated run; only the operand payload and driver differ.
Pinned Spike matched all **1,024 padded BF16 outputs** against the
independent Radiance arithmetic reference. The 512 real outputs have mean
absolute error **0.02569** and maximum absolute error **0.16851** relative
to the captured PyTorch f32 `softmax @ V` output. These are diagnostics for
this reduced random-weight model, not model-quality thresholds. Two clean
Spike runs reproduced 15 stable artifact hashes; the only receipt difference
was a path-bearing linker log hash. See `reproduction.json`.

`bundle/manifest.json` names the upstream softmax summary, VPU probability
bytes, QK golden bytes, full model graph, and source regions by digest. The
PV qualifier checks those against the exact selected model head and V tensor
before quantizing the PV operands. `source/` contains the floating source
probabilities, V tensor, and capture receipt. The packed payload and physical
command program are in `bundle/` and `object/`.

**Execution scope:** QK, VPU softmax, and PV ran as three separate Spike
programs. The host applied scale and mask, decoded the VPU result, and staged
each next input. The compiler does not yet route the score and probability
buffers on device, lower the entire attention graph, or run full TinyLlama.
Nicolas's [current application kernels](https://github.com/ucb-bar/npu-exploration/tree/main/baremetal/mxgemmini/src)
also include requantized `P @ V` resident in scratchpad before `o_proj`,
while this archive checks BF16 PV readout. That resident path is a separate
coverage target.

Check the saved evidence with:

```sh
python -m pytest -q tests/test_tinyllama_connected_attention_head_evidence.py
```

Regenerate by preparing the TinyLlama PV inputs with
`tools.prepare_tinyllama_attention_head_inputs --stage pv`, then running
`tools.qualify_model_attention_head --stage pv --site-region matmul_5` with
the archived full model graph, `source/value.npy`, the VPU BF16 probability
file, softmax summary, QK bundle, PV input receipt, and the pinned MX+VPU
profile. Pass those four producer inputs with
`--upstream-softmax-summary`, `--upstream-probability-bf16`,
`--upstream-qk-bundle`, and `--source-pv-input-receipt`. The activation NPY
must decode the exact BF16 probability bytes; the qualifier rejects any
changed source, profile, head, or V tensor.
