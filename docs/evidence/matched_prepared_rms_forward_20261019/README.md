# Complete TinyLlama forward: BF16 broadcast preparation

Preparing repeated broadcast operands as BF16 reduces the complete independent
160-token, 22-layer forward from **13,479,416,598 to 13,215,457,143 modeled
cycles**, a further **1.9582% reduction**. Every one of the **5,120,000 output
words** remains identical, SHA-256
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.

The candidate is **17.3241% faster** than the corrected quality-preserving
compiler baseline (15,984,649,026 cycles) and **3.5503% slower** than Nicolás's
matched token-to-logits control (12,762,360,525 cycles). The unchanged quality
gate passes on the 33 retained full source logit rows: mean KL 0.06379562704,
worst KL 0.16997975795, and top-token agreement 27/33. The gate retains the
original 5% KL margins and top-token requirement. Full compiler FP32 top-token
agreement is 133/160; full source distributions for all 160 positions are not
available. This result does not yet beat the source implementation.

Both builds include the 45 connected broadcast multiply pairs documented in
[matched_rms_forward_20261019](../matched_rms_forward_20261019/README.md).
The only new selected optimization is `--prepare-vpu-broadcast-operands`, with
the original 1 MiB per-adapter preparation budget. Merlin plans invariant
broadcast conversions from input shapes, axes and the byte budget. MX uses
prepared BF16 input views while preserving intermediate rounding and exceptional
value refusal. The section evidence and retained regressions are in
[broadcast_multiply_20261019](../broadcast_multiply_20261019/README.md).

This full comparison does not include integer group packing, guarded host math
or movement-based native tiling. Those are distinct full-model experiments.
Both actual processes exited zero with `DONE`, complete output words, accepted
structural ELF admissions, uninstrumented timers and identical checkpoint/input
identities, profile, extension and simulator settings.

Whole-process counters are identical: 3,312,204 commands, 38,608 native loops,
492,001 fences, 29,364 VPU commands, 40,970,240 mesh tiles, 655,523,840 mesh busy
cycles, 4,198,111,552 load bytes and 115,279,872 scale bytes. The optimization
reduces repeated host conversion/indexing work. Counter scope is wider than
the stage timer, and host-memory tracking is off.

The linked ELF grows by 64 bytes, 1,330,818,608 to 1,330,818,672. Literal
allocation-site sums remain 1,372,484,654 bytes at 1,061 sites, with zero dynamic
sites; these are not peak memory. Both use the original 2 GiB arena. The exact
14,496-byte target quantizer object is retained and identical in both images,
SHA-256 `f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0`.
The archived historical C/header/LUT snapshot is distinct from the later generic
word-packing revision. Scalar `runtime_compiler` receipts describe MLIR support,
not this target quantizer object.

```sh
python docs/evidence/matched_prepared_rms_forward_20261019/verify.py
# Additionally reopen the actual external linked images:
python docs/evidence/matched_prepared_rms_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone verifier needs Python and NumPy, imports no compiler package,
and uses the pinned adjacent full-log reader and retained independent-source
archives. It decodes every compiler word, checks full-output hashes and cycle
records, recomputes the unchanged source-relative gate, checks raw counters,
and binds the original build, terminal, recipe, admission and live-environment
receipts. The two full logs, generated target sources, transformed IR, catalog
and rewrite records, original job specifications/status, capture IR/input
identities and quantizer objects are retained. Every member is pinned with its
original workspace path in `artifacts.json`.

Large model ELFs, packed checkpoint tensors/objects and transitive toolchain
inputs remain external. This is compact measurement custody, not a hermetic
rebuild or offline admission replay. Exact original commands are in
`baseline/job/spec.json` and `candidate/job/spec.json`. Reproduce in fresh
task-local workdirs with unchanged capture, numerical policy, checkpoint reuse,
profile, extension, `both/e4m3_vpu/mem.host_tracking=0` settings, original timing
boundary and complete dump selection. Run the independent complete-output
validator, then `tools.compare_matched_tinyllama_forward` with the existing
same-input source and token-lookup boundary archives.
