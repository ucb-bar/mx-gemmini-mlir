# Independent 128-token TinyLlama prefill diagnostic

This bundle checks one 128-token prefill through all 22 TinyLlama decoder
layers and the LM head. Nicolas's `llama_model.c` source and the compiler's
model2MLIR → Merlin → MX+VPU program use the same 128 input IDs and the same
200 FP32 checkpoint tensors. The source input file includes one additional
next-token **label** used by its generator; that ID is not a forward input.
The [weight receipt](weight_receipt.json) and the two input files pin the
identity checks.

This is an **all-FP8 diagnostic**. The compiler's
[override recipe](all_fp8_overrides.json) promotes 155 matrix sites from the
captured mixed FP4/FP8 policy to FP8, giving 200 FP8 MX bindings in the
[catalog summary](catalog_summary.json). It also offloads 88 add regions to
the VPU. The source calls its VPU residual-add path 44 times. Nicolas's
source is adapted with a 64-row Q/K/V and LM-head tile in
[this patch](m128_tiling.patch) to respect its activation scale window.
The pinned source accepts that zero-context patch with
`git apply --unidiff-zero` after the archived VPU residual patch.

Both executables run on Nicolas's pinned `e4m3_vpu` Spike extension with
`GEMMINI_MODE=both`, `GEMMINI_PERF_SET=mem.host_tracking=0`, and the
`rv64gc_zicntr` ISA. The source counter starts before `gemmini_flush` and
ends after the final head fence. The compiler's two `cycle` CSR reads enclose
the session reset through state commit. The archived verifier checks all
32,000 final logits, the source output against the generated BF16 reference,
and the two modeled cycle counters. See [receipt.json](receipt.json) after
the completed runs, then run:

```sh
python3 -B docs/evidence/nicolas_tinyllama_prefill128_source_20261014/verify.py
```

| Program | Modeled stage cycles | Logit cosine vs captured FP32 | Top token |
| --- | ---: | ---: | ---: |
| Nicolas source | 9,611,186,417 | 0.9932 | 29871 |
| Compiler, 200 FP8 MX sites | 11,442,665,332 | 0.9432 | 362 |

The two programs' final logits have cosine **0.9454** to each other; the
captured FP32 top token is **279**. The source's emitted final BF16 logits
match its generated quantized reference exactly. These measurements show a
substantial numerical gap in the compiler prefill for this input, alongside
the separate mixed FP4/FP8 [precision diagnostic](../tinyllama_prefill_precision_20261014/README.md).

The raw cycles have **no speedup ratio**. The source computes 128 valid
attention keys while the compiler's first QK operation issues 160 allocated
key slots. Host math, VPU work, and timing setup also differ. This stage has
neither the 32 decode steps nor the mixed FP4/FP8 policy of the complete
captured TinyLlama session; it is an independent source-program diagnostic,
not a full-model handwritten baseline. The full SmolVLA issuer-swap result
likewise remains a separate comparison with fixed host and VPU objects.
