# Complete TinyLlama forward: guarded host expf

The explicitly selected guarded host-math route reduces the independent
160-token, 22-layer token-to-logits forward from **11,197,470,681 to
11,105,781,721 modeled cycles**, a further **0.8188% reduction**. The candidate
uses **12.9802% fewer cycles than the reference implementation’s matched 12,762,360,525-cycle control**
and **30.5222% fewer** than the corrected quality-preserving compiler baseline
of 15,984,649,026 cycles.

All **5,120,000 compiler logit words** remain byte-identical, SHA-256
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
The unchanged reference-relative gate passes on the 33 archived full source
distributions: mean KL 0.06379562704, worst KL 0.16997975795, and FP32 top-token
agreement 27/33. The original 5% KL margins and top-token requirement remain
unchanged. Complete compiler FP32 top-token agreement is 133/160. Full source
distributions for all 160 positions were not retained.

The host-math contract selects `expf` → `__ieee754_expf` only for finite
arguments with magnitude at most 80, and exact negative infinity → positive
zero. Other values retain the original call. The contract pins the actual libm
archive and requires identical return bits, errno and floating effects across
supported rounding modes. This is not a blanket replacement or a fast-math
permission. Generic guarded-call lowering belongs in Merlin; target toolchain
endpoint selection and library qualification remain explicit.

The source/binary review, 1,966,080 paired native value/effect comparisons, and
retained out-of-domain fallback regression are in
[guarded_host_math_20261019](../guarded_host_math_20261019/README.md). Those
section observations are distinct from this complete uninstrumented model
measurement. No full-model guard-frequency estimate is inferred from them.

The two runs use identical captures, checkpoint tensors, numerical placements,
target profile, simulator binary/extension/settings and logical timing
boundaries. Both actual processes exited zero, emitted complete logits and have
accepted structural ELF admissions. The source control includes token embedding
lookup. This frozen build does not include movement-based native tiling or the
new VPU-to-matrix BF16 transport; those have separate full-model experiments.

Whole-process counters are unchanged: 3,312,204 commands, 38,608 native loops,
492,001 fences, 29,364 VPU commands, 40,970,240 mesh tiles, 655,523,840 mesh busy
cycles, 4,198,111,552 load bytes and 115,279,872 scale bytes. These counters have
a wider boundary than the stage timer; host-memory tracking is off.

The ELF grows by 96 bytes, 1,330,818,584→1,330,818,680, with candidate SHA-256
`5c5d1b855be855d63cff66e7ff601a284107720ca757b4cca971edefc52723ff`.
Literal allocation-site sums remain 1,372,484,654 bytes at 1,061 sites, zero
dynamic sites, and the original 2 GiB arena. These are not peak memory. Both
target quantizer objects remain byte-identical at 12,656 bytes, SHA-256
`edc9fd2f4a600078d02d7ccd5dd553042b7cdda91094cedfd692bd424fc1cda8`.
The exact guard contract, host-math C/objects and quantizer snapshots are retained.

```sh
python docs/evidence/matched_guarded_math_forward_20261019/verify.py
# Additionally reopen the retained external full model images:
python docs/evidence/matched_guarded_math_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone verifier uses Python, NumPy and pinned adjacent full-word/source
evidence, with no compiler imports. It decodes every compiler word, recomputes
the unchanged quality gate, checks the source timer and native counters, and
binds original run/build/recipe/admission/environment records and quantizer
source identities. Full logs, capture IR/input identities, generated target
code, transformed IR, exact small objects/source snapshots and original runner
commands/status are archived. Large full ELFs, packed checkpoint data/objects,
the libm archive and transitive toolchain inputs remain external. This is not a
hermetic rebuild or offline replay of admission. Reproduce from the retained
`job/spec.json` commands in fresh task-local workdirs with the original inputs,
guard contract, extension, timing settings, complete output cap and unchanged
token-to-logits boundary, then run the completed-output validator and matched
source-relative comparator. Newer source builds are separate experiments.
