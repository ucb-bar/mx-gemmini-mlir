# SmolVLA source-attention cutpoint

This cutpoint comes from the pinned `lerobot/smolvla_base` checkpoint at
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`. The flow-stage inputs
come from the model2MLIR session capture, not from a synthetic shape-only
example. The input NPZ SHA-256 is
`4ddbc494f37c31c4bba97f7158f73e90d8491ab40b20fc1536626e3319a57075`.

The included [`cutpoint.npz`](cutpoint.npz) records the first attention call
in the first denoising step: `q[15,50,64]`, `kt[15,64,163]`, probabilities
`p[15,50,163]`, `v[15,163,64]`, and the Boolean key mask
`mask[1,1,50,163]`. Its SHA-256 is
`6b5b713660c819aa8f69a5174464b46258491bc81a86a8b4073b6c56cc9f122a`.
The mask is 84.9693% true and has no fully masked query row. Run
`python -B verify.py` with NumPy to check that masked `softmax(q @ kt / 8)`
matches the captured probabilities; the RMSE is below `2e-9`.

To recapture with the pinned checkpoint already cached locally, run from this
directory (substitute your model2MLIR checkout and captured stage paths):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B capture.py \
  --model2mlir "$MODEL2MLIR" \
  --flow-stage "$CAPTURE/stages/flow_denoise" \
  --out cutpoint.npz
python -B verify.py
```

The capture hook changed no eager output: tapped and untapped flow-step
outputs were bit equal. The newly loaded eager model's first-step output
differs from the saved model2MLIR FP32 golden by `0.00054054` RMSE. This
drift is recorded rather than presented as exact capture parity. The saved
golden NPZ SHA-256 is
`d61259c6e424d41767efded955832af1e58b0ed5358b7803854d8d4b0435ac3b`.

This provides real operands and a reference for adapting Nicolas's
[`attn_flash.c`](https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/src/attn_flash.c)
to SmolVLA. That source currently implements dense or causal attention, not
this arbitrary Boolean mask.

## One-head source run

The [small source patch](attn_flash_smol.patch) adds a block-major BF16
additive mask to Nicolas's existing online softmax. The
[generated operand header](smol_head0.h) uses his E4M3 quantizer on the
captured first head, pads 50→64 query rows and 163→192 keys, and masks
padded keys before VPU row maximum. Padded query rows are discarded. The
patch leaves the existing MX/VPU command schedule intact.

With Nicolas's `e4m3_vpu` Spike timing extension and `GEMMINI_MODE=both`,
the [source log](source_spike.log) reports:

| First-head pass | Modeled cycles | Output check |
| --- | ---: | --- |
| Serial diagnostic | 33,961 | `437d801250f000dd` |
| Pipelined | **22,447** | Same hash; `attn_flash PASSED` |

The BF16 output's relative Frobenius error versus captured FP32 `p @ v`
over the 50 valid query rows is **3.4425%**. The serial and pipelined
outputs are bit identical. The [receipt](source_receipt.json) pins the source,
patch, cutpoint, generated header, ELF, Spike extension, and log hashes.
The archived [generator](generate_header.py) reproduces the operand header:

```sh
python -B generate_header.py \
  --rocc-tools "$ROCC_ROOT" --out "$OUT_DIR/smol_head0.h"
```

`ROCC_ROOT` here is the directory containing `gen_matmul_llama.py`; its
sibling `npu-exploration` checkout provides that generator's helpers. To
rebuild and run the archived header, set `NPU_ROOT` to pinned
`npu-exploration` `0344bcb`, `ROCC_ROOT` to the matching
`gemmini-rocc-tests` directory, `RISCV_CC`, `SPIKE`, `SPIKE_EXTENSION`, and
`OUT_DIR`, then run `bash run_source.sh`. Set `RISCV_ENV` separately if the
ROCC checkout's `riscv-tests/env` directory is incomplete.

This is **one attention head in one flow step**. It is not a matched compiler
cutpoint, a full SmolVLA run, or a stitched-model cycle ratio.
