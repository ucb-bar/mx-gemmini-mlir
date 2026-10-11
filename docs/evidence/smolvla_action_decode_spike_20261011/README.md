# SmolVLA action-decode host stage on pinned Spike

The full-checkpoint model2MLIR `action_decode` stage contains no direct MX
contraction. The selected MX+VPU stage runner therefore compiled it through
Merlin's RV64 host path, recorded zero MX selections, and ran the resulting
ELF on Nicolas's pinned Spike. The [raw console](spike.log) contains all 1,600
f32 output values, their typed and indexed digests, `DONE`, and zero memref
rank mismatches. Its complete output matches the capture's saved PyTorch
`golden.npy` bit for bit.

| Item | Recorded value |
| --- | --- |
| model2MLIR stage SHA256 | `c194700d5a9e19aa83456862f666a41559c83e51b992709bba598d681176224b` |
| MX+VPU profile SHA256 | `e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d` |
| Spike Gemmini extension SHA256 | `5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5` |
| Merlin build hash | `ae56bbde5e18` |
| f32 result storage SHA256 | `c136c05e1e99b3f9013802ae214f5b9e0ee5c57735f401cdcfcb508dd5695c6f` |
| saved PyTorch `golden.npy` file SHA256 | `9199f2280fd2f754cdfd42779cccf77d99004089c9001cd55292dd473e0b0959` |
| raw console SHA256 | `a2c40bd62d005e93ad7b8a9f1eb193cd66c2699b989784f37ede7f0f3e3f2410` |

The input is the captured action-decode stage input. This run does not route
the output of a compiled flow-denoise stage into action decode, and it does not
establish end-to-end SmolVLA inference or MX numerical correctness.
