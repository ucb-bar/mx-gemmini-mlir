# Two-layer TinyLlama whole-stage Spike diagnostic

This is a **random-weight diagnostic**, not a pretrained TinyLlama inference
result or a quantized PyTorch parity certificate. A receipt-bound model2MLIR
prefill graph compiled through Merlin with 19 source-bound MX calls and seven
unique FP4/FP8 kernels. Nicolas's pinned Gemmini Spike extension ran the
single-stage ELF to `DONE` with a 2 GiB model arena. The raw console is in
[`spike.log`](spike.log).

| Bound item | Digest or count |
| --- | --- |
| model2MLIR `model.mlir` SHA256 | `78980b12f4696c0007e8d079b2d57a44b77821141dfe586d490462565f66bd6b` |
| MX+VPU profile SHA256 | `e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d` |
| Nicolas Spike extension SHA256 | `5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5` |
| Merlin build hash | `85d13340de9e` |
| Routed MX calls | 19 |
| Unique kernels | 7 |
| Model arena | 2 GiB |

The full first-output record covers 32,000 f32 logits. The target output
SHA256 is `756897f97073152ef0cc3eba4ae09be41a2295d688ebf3816dcdaaf52a6b12aa`.
The original unquantized PyTorch `golden.npy` has a different digest and
argmax (21,282 versus target 26,705). FP4/FP8 conversion changes the program;
an independently quantized PyTorch reference is required to judge the target
result. This run exercised one prefill stage, not decode or recurrent state.
