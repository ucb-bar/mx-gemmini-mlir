# Native gpt-oss expert tile admission

`tools/bind_gptoss_native_tile.py` admits one range-extracted gpt-oss expert tile
without decoding or requantizing its checkpoint MXFP4 weights. Its input
receipt is `mx_checkpoint_ranges.v1` from the checkpoint range extractor. The
tool verifies the pinned revision, the full original tensor names, the selected
expert index and shapes, inclusive source byte ranges, exact extracted payload
hashes, and the reverse mapping from Gemmini B bytes to checkpoint bytes.

The `--activation-bf16` input is an explicit row-major BF16 `[M,K]` operand,
where `K = 32 * --block-count`. It is copied into the output with its SHA-256.
The output contains `a_bf16.bin`, `b_codes.bin`, `b_scales.bin`, and
`manifest.json`. B codes are `[K,N/2]`, pairing adjacent output columns in
each byte; B scales are E8M0 `[K/32,N]`. Coordinates in the manifest refer to
the original expert's output rows and input MX groups.

For the verified gpt-oss-20b layer 0 expert 0 gate/up fixture:

```sh
python3 tools/bind_gptoss_native_tile.py \
  --fixture-dir "$TASK_DIR/.tmp/gptoss20b-expert0" \
  --extractor-receipt "$TASK_DIR/.tmp/gptoss20b-expert0/receipt.json" \
  --expected-revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --activation-bf16 "$TASK_DIR/.tmp/gptoss20b-expert0/a_bf16.bin" \
  --activation-m 1 --projection gate_up --expert-index 0 \
  --output-start 0 --output-count 32 --block-start 0 --block-count 2 \
  --output-dir "$TASK_DIR/.tmp/gptoss20b-native-tile"
```

Use the actual receipt filename emitted by the extractor. For this example A
must have 128 bytes. `--profile` can additionally require that a source-bound
target profile contains the direct FP4 A/B mesh tuple. The receipt is a trusted
input from the checkpoint extractor; the manifest records its hash so a later
measurement can pin the exact admission evidence.

The status and `execution_refusal` in `manifest.json` are deliberate: direct
FP4 mesh execution needs packed MXFP4 A codes and E8M0 scales. This slice
supplies BF16 A bytes without choosing an activation quantization policy, and
the backend has no direct BF16 by checkpoint-MXFP4 GEMM lowering. It therefore
emits no executable and supplies no Spike, FireSim, numerical, or performance
claim. The user-selected recipe must define the activation policy before that
next step can be admitted.
