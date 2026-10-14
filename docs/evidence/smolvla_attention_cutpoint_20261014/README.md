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
matches the captured probabilities; the RMSE is below `5e-9`.

To recapture with the pinned checkpoint already cached locally, run from this
directory (substitute your model2MLIR checkout and captured stage paths):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B capture.py \
  --model2mlir "$MODEL2MLIR" \
  --flow-stage "$CAPTURE/stages/flow_denoise" \
  --out cutpoint.npz
python -B verify.py
```

The same flow step also has attention over 113 prefix keys. Capture its
first 15-head call with `capture.py --keys 113 --out cutpoint113.npz`.
Its archived [`cutpoint113.npz`](cutpoint113.npz) has SHA-256
`19fcd4c7ec2e39870632ad0b97074b240029afd77c78b98bf8c148eae338f356`.
All 113 logical keys are valid; masked padded keys still matter when it is
mapped to 128 physical columns. Its recomputed softmax has RMSE below
`3e-8` against captured probabilities across the NumPy builds used here.

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

The [all-head receipt](source_all_heads.json) and
[15 source logs](source_all_heads/) extend the same check to every captured
head in that first flow step. All 15 serial/pipelined output hashes agree
and all pass the source checker. Each separately timed pipelined run takes
22,447 modeled cycles with this fixed shape and schedule. Relative error
versus each head's captured FP32 output ranges from **2.977% to 6.356%**.
The 336,705-cycle sum is a diagnostic sum of 15 standalone passes.

The [113-key receipt](source_113_all_heads.json) and
[15 more source logs](source_113_all_heads/) cover the other attention shape.
These heads use 64×128×64 physical tiles with an archived
[head-0 operand header](smol_head113.h). All 15 pass the source checker;
their separately timed pipelined passes take **16,775 cycles each** and
their relative errors range from **3.592% to 7.751%**. The 251,625-cycle
sum is another diagnostic sum. Neither sum is a stitched stage timing.

This regenerates the first shape using the same
environment variables as `run_source.sh`; `ROCC_TOOLS` can name a separate
quantizer-tools checkout:

```sh
python -B run_all_heads.py --out-dir "$OUT_DIR"
python -B run_all_heads.py --cutpoint cutpoint113.npz \
  --out-dir "$OUT_DIR/keys113"
```

These are **attention heads in one flow step**. They are not matched compiler
cutpoints, a full SmolVLA run, or a stitched-model cycle ratio.

## Flow-stage attention roster

The [projected flow catalog](flow_attention_catalog.json) records the 32
QK/PV matrix bindings from the corrected mixed FP4/FP8 full-session flow
build. Eight attention calls
per step have 163 keys and eight have 113 keys; each has 15 heads. The
[roster analysis](flow_attention_roster.json) combines those counts with the
separately timed first-call source heads. Repeating their warm head timings
arithmetically gives 4,706,640 cycles for the matrix and VPU work represented
by 16 calls in one step, or 47,066,400 over ten steps. This is an
**illustrative sum**, not a measured stage or model timing. It omits all
other flow operations, host work, and scheduling between calls, and reuses
first-call operands and timings for later calls and steps. It cannot support
a compiler-versus-source speedup claim.

To regenerate the clean catalog projection from a captured flow build, pass
its `device_catalog/mx_catalog.json` and `mx_stage_probe.json` to
`project_flow_catalog.py --catalog ... --probe ...`, then run
`python -B analyze_flow_attention_roster.py`.
