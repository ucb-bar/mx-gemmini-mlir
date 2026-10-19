# Complete TinyLlama forward: generic Merlin word-group packing

The generic word-group implementation completes the independent 160-token,
22-layer token-to-logits forward in **11,197,470,681 modeled cycles**:

- **12.2618% fewer cycles** than the reference implementation’s matched control, 12,762,360,525.
- **29.9486% fewer cycles** than the corrected quality-preserving compiler
  baseline, 15,984,649,026.
- **15.2699% fewer cycles** than its matched prepared-broadcast control,
  13,215,457,143.

All **5,120,000 output words** remain bit-identical to that compiler control,
SHA-256 `d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
The unchanged reference-relative gate passes on the **33 archived complete source
logit distributions**: compiler mean KL 0.06379562704, worst KL 0.16997975795,
and FP32 top-token agreement 27/33. Source values are 0.06417128458,
0.18960313411 and 27/33. The original 5% KL margins and top-token requirement
remain unchanged. Complete compiler FP32 top-token agreement is 133/160; full
source distributions for all 160 positions were not archived.

Merlin owns the serial strided word preparation and maximum-publication
mechanism. MX supplies BF16 rounding, finite membership, scale/codebook/layout
and native compilation. `--integer-quant-groups` selects call-local uint16_t
groups and fixed-axis inlining. The old implementation already cached rounded
floats; it did not round every element twice. This reduces group representation
and preparation costs without changing the numerical policy.

This is the **generic Merlin implementation**, distinct from the historical
[OOT prototype](../matched_word_packing_prototype_20261019/README.md).
It is a frozen measured build, before the subsequent BF16-input refactor and
reverse transport work. Guarded host math, movement-based native tiling and
VPU-to-matrix BF16 transport are not included in this measurement. Those have
separate full-model experiments. Do not attribute this ELF's exact timing or
object identity to a later rebuilt compiler revision.

Both complete processes exited zero, have accepted structural ELF admissions
and uninstrumented timers, and use identical input/checkpoint tensors, target
profile, simulator binary/extension/settings and logical timing boundaries.
The matched source timer includes token embedding lookup, rather than starting
from precomputed embeddings. The older 15.171B compiler baseline has different
quality and is not an equal-quality speedup control.

Whole-process counters are unchanged: 3,312,204 commands, 38,608 native loops,
492,001 fences, 29,364 VPU commands, 40,970,240 mesh tiles, 655,523,840 mesh busy
cycles, 4,198,111,552 load bytes and 115,279,872 scale bytes. These counters have
a wider boundary than the stage timer; host-memory tracking is off. The gain is
in host packing work, without a reduction in native commands.

The full ELF is 1,330,818,584 bytes, SHA-256
`12e08dcca5add8efe17be17d193c3911e4f7f49b027ec459e41f9926f0558003`,
88 bytes smaller than the matched control. Literal allocation-site sums remain
1,372,484,654 bytes at 1,061 sites, with zero dynamic sites. These are not peak
memory. Both retain the original 2 GiB arena. The target quantizer object is
12,656 bytes, SHA-256
`edc9fd2f4a600078d02d7ccd5dd553042b7cdda91094cedfd692bd424fc1cda8`.
The exact C/header/LUT and generated Merlin preparation header are retained.
The complete section evidence, including refusal cases and stack observations,
is in [group_word_packing_20261019](../group_word_packing_20261019/README.md).

```sh
python docs/evidence/matched_merlin_word_packing_20261019/verify.py
# Also reopen the retained external complete model images:
python docs/evidence/matched_merlin_word_packing_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone verifier requires Python, NumPy and digest-pinned adjacent
full-word/source evidence, with no compiler imports. It decodes all compiler
words, recomputes the original quality gate, checks the equal-boundary source
timer and native counters, and binds completed build/validation/recipe/admission
receipts, live process environments and historical quantizer source hashes.
The compact archive retains full logs, capture IR/input identities, generated
target code and transformed IR, catalog and rewrite records, exact quantizer
objects/source/header snapshots and original runner commands/status. Every
member is pinned with its original path in `artifacts.json`.

Large model ELFs, packed checkpoint tensors/objects and transitive toolchain
inputs remain external. Receipt checks are not a hermetic rebuild or offline
replay of admission. Original complete commands are in each `job/spec.json`.
Reproduce in fresh task-local workdirs with unchanged capture, checkpoint reuse,
numerical policy, profile, extension, `both/e4m3_vpu/mem.host_tracking=0`
settings, original token-to-logits boundary and complete dump selection, then
run the complete-output validator and matched-forward comparator. Rebuilding
newer source produces a new experiment rather than these exact ELF hashes.
