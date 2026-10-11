# Full-checkpoint TinyLlama prefill on Nicolas's MX Spike

The full-checkpoint eager-attention TinyLlama prefill compiled through
model2MLIR, Merlin, and the MX dense catalog. A manual relink of the same
stage objects with a 5 GiB model arena ran on Nicolas's pinned Gemmini Spike
extension and reached guest `DONE` after 5,288.408 seconds (about 88 minutes).
The [raw console](spike.log) records 374,944,319,266 simulated cycles, build
hash `845c3a5af7bd`, and zero memref rank mismatches. The process exited zero.

| Bound item | Value |
| --- | --- |
| Linked session's captured eager prefill MLIR SHA-256 | `8b7e6e709fd12352773b5a7b264916b3f98b9a6b901f329de04446619ef09031` |
| Stage build's prepared MLIR SHA-256 | `060e7efacf91b8eb1e3829e903697ab4b246729ede7055c504b182f13b260543` |
| MX catalog source SHA-256 | `6c1cf88337d770a7865b9d4aa0de2b55903e9892edd6d015fb1640c632e253d4` |
| MX+VPU profile SHA-256 | `e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d` |
| Relinked stage ELF SHA-256 | `6bb09b65aabf54e4346a03afcc7481893a53640c02b50dd43f75ebd8bbbf869a` |
| Nicolas Spike executable SHA-256 | `de7573f938268f77b8a1729420d3c47aacebed571efc090da73ee87078ce01e9` |
| Gemmini extension SHA-256 | `5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5` |
| Raw console SHA-256 | `3d87b79641849bc67f6518f91a77d5985b8c341b526497d095678fcc24b1e3ef` |
| Routed MX calls | 199 |
| Full first-output size | 32,000 f32 logits (128,000 bytes) |
| First-output SHA-256 | `e829ba982b701c5c8e2b64af8d9279cd23a106ddfa84ca147d6c82a00f4b6459` |
| Target top token | 1113 |

The [independently linked full TinyLlama session](session_prefill_prefix.log) emitted that exact
128,000-byte first-output digest for prefill program 0, step 0, output 0. Its
separate console also contains digests for the two prefill state results and
has entered decode step 0. This agreement checks the first-output handoff
between the standalone stage and session links. The linked 32-step session
has not reached guest `DONE`.

This is an **execution and cross-link consistency result**, not an inference
accuracy result. The captured unquantized PyTorch prefill chooses token 279;
the BF16 PyTorch operand fake-quant diagnostic chooses token 29871. That proxy
does not reproduce the MX mesh arithmetic and cannot judge the Spike token.
An independent quantized reference for every output and recurrent state is
still required. The 5 GiB stage ELF was manually relinked, so this archive
does not claim the usual automated stage receipt gate for that ELF.
