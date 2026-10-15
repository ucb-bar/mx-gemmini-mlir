# TinyLlama token-to-logits source control

This archive compares one 160-token causal forward through all 22 TinyLlama
decoder layers and the LM head. One program is Nicolas's independent MX+VPU
source schedule; the other is the model2MLIR → Merlin → MX+VPU compiler output.
Both receive the same 160 token IDs, the same 200 FP32 checkpoint tensors,
and all-FP8 matrix precision on the same Nicolas Spike extension. The
[earlier input archive](../nicolas_tinyllama_onepass160_same_input_20261014/README.md)
records the capture, revisions, and pinned target profile.

The independent source originally began with precomputed BF16 embeddings.
`tools/prepare_nicolas_tinyllama_lookup.py` adds its checkpoint FP32 token
table and gathers those IDs **inside** the original source stage timer. All
327,680 selected BF16 embedding values equal the previous source input. The
source generator's payload and header must match the earlier archive byte for
byte. This gives both programs a token-ID-to-logits logical timing boundary.
The matrix and host schedule remains Nicolas's patched `llama_model.c` from
that archive; the new source edits are the token lookup and an untimed
full-output argmax print.
The source and compiler still use different host/VPU arithmetic and command
schedules, so their output quality must accompany their raw cycle counts.

| One 160-token forward | Nicolas source | Compiler |
| --- | ---: | ---: |
| Spike modeled cycles, token IDs to logits | 12,762,360,525 | 15,170,996,572 |
| Top tokens matching captured FP32 | 131/160 | 103/160 |
| Late top tokens matching captured FP32 | 27/33 | 20/33 |

The compiler uses **18.873% more modeled cycles** in these measured
programs. The programs select the same top token at 114/160 positions
(24/33 late positions), so the cycle difference is a **raw comparison**,
not an equal-quality speedup. The independent source's 160 top tokens
match the generated BF16 reference exactly. Its 33 emitted BF16 logit rows
also match the earlier source run exactly. Adding the lookup to the source
timer adds 58,646,741 modeled cycles over the earlier precomputed-embedding
run without changing those logits.

The source and compiler each execute about 40.97 million modeled mesh tiles,
but the source issues 880 VPU commands and loads 2.713 GB through the
extension, while the compiler issues 44,872 VPU commands and loads 4.408 GB.
These are [source](source_perf.txt) and
[compiler](../nicolas_tinyllama_onepass160_same_input_20261014/compiler_perf.txt)
extension counters. They show different vector and data movement policies;
they do not isolate a single cause of the cycle or numerical difference.

The archive keeps the lookup, build, and Spike run receipts, a compressed
source log, and `fair_comparison.json`. The source payload, FP32 embedding
table, object files, and executable are large task-local artifacts; their
hashes are in the receipts. Run `python -B verify.py` here to check the
portable evidence against the earlier source and compiler archive.

This is one full-depth forward, not the 128-token prefill plus 32 cached
decode calls. The full-session issuer-only controls are documented in
[the comparison guide](../../fair_full_model_comparisons.md).

## Reproduce

Regenerate the source payload and 160-token capture as described in the
[earlier input archive](../nicolas_tinyllama_onepass160_same_input_20261014/README.md).
Apply the same three source patches from that archive before preparing this
control. Set `SOURCE_DIR` to the patched C file and generated header/blob,
`ACT_DIR` to the matched Nicolas activation capture, and `BUNDLE` to the
model2MLIR compiler capture. Run the following from the repository root:

```sh
python -B -m tools.prepare_nicolas_tinyllama_lookup \
  --source-c "$SOURCE_DIR/llama_model.c" \
  --source-activations "$ACT_DIR/model.npz" \
  --compiler-bundle "$BUNDLE" --out-dir "$PREPARED"

python -B -m tools.run_nicolas_tinyllama_lookup \
  --prepared "$PREPARED" \
  --source-blob "$SOURCE_DIR/llama_model_m160.bin" \
  --source-header-dir "$SOURCE_DIR" \
  --prior-comparison \
    docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014/matched_onepass_comparison.json \
  --npu-root "$NPU" --rtl-root "$RTL" --riscv-root "$RISCV" \
  --extension "$EXTENSION" --out-dir "$PREPARED/run"
```

Use the pinned `npu-exploration`, `gemmini-mx-cleanup`, RV64 toolchain, and
Spike extension revisions in the earlier archive. The runner checks the
generated source payload and header hashes before building and records the
exact command, executable hash, extension hash, and cycle counter.
