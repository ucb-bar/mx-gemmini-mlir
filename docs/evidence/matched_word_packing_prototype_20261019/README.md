# Historical word-packing prototype: complete TinyLlama forward

The earlier word-group packing prototype completed the same independent
160-token, 22-layer token-to-logits forward in **11,281,937,006 modeled cycles**,
versus **13,215,457,143** for its prepared-broadcast control: **14.6307% fewer
cycles**. It uses **11.5999% fewer cycles** than Nicolás's matched token-to-logits
control (12,762,360,525).

This is the **earlier OOT prototype**, not the later generic Merlin word-group
implementation. Their compiled quantizer objects differ. The actual executed
prototype C/header/LUT and quantizer object are retained and source-bound to its
catalog. Do not attribute this measurement to the final generic implementation;
that is a separate complete-forward experiment.

All **5,120,000 compiler output words** are byte-identical to the matched control,
SHA-256 `d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
The unchanged Nicolás-relative quality gate passes on the 33 archived complete
source distributions: mean KL 0.06379562704, worst KL 0.16997975795, and FP32
top-token agreement 27/33. The original 5% KL margin and top-token requirement
remain unchanged. Compiler FP32 top-token agreement across all 160 positions is
133/160. Full source distributions for all 160 positions were not retained.

Both runs are uninstrumented, completed successfully, use identical captures,
checkpoint tensors, target profile, simulator/extension settings and logical
timing boundaries, and have accepted structural ELF admissions. The prototype
changes group preparation and inlining without changing MX numerical policy,
block scales, codebook, layout or native matrix/VPU placement. The old group
implementation already cached rounded floats; this is not removal of an alleged
second rounding pass. The later generic mechanism and section comparisons are
documented in [group_word_packing_20261019](../group_word_packing_20261019/README.md).

Whole-process counters remain 3,312,204 commands, 38,608 loops, 492,001 fences,
29,364 VPU commands, 40,970,240 mesh tiles, 655,523,840 mesh busy cycles,
4,198,111,552 load bytes and 115,279,872 scale bytes. Host tracking is off and
these counters cover a wider boundary than the stage timer. The speedup comes
from host packing work, without fewer native commands.

The full ELF shrinks from 1,330,818,672 to 1,330,818,584 bytes. Literal allocation
site sums remain 1,372,484,654 bytes at 1,061 sites, with zero dynamic sites;
these are not peak memory. The original 2 GiB arena is retained. The actual
prototype quantizer object is 12,920 bytes, SHA-256
`225ff337495c18dc5b9eff035848bf543d9016e51c666d31966e47f9e2ba3496`,
versus the control's 14,496-byte object. The prototype C SHA-256 is
`8424afb9a96e0bee09e7b3098fe97fab1008b64e84f5464b1be1b1c9fe560e4a`.

```sh
python docs/evidence/matched_word_packing_prototype_20261019/verify.py
# Also reopen the original external full ELFs:
python docs/evidence/matched_word_packing_prototype_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The verifier uses Python, NumPy and the pinned adjacent full-word reader and
independent-source archives, with no compiler imports. It decodes every compiler
word, recomputes the unchanged quality gate, checks the equal-boundary source
timer and native counters, binds completed recipes/admissions/receipts and live
environment observations, and checks the historical quantizer source hashes.
The archive retains full logs, capture IR/input identities, rewritten IR,
generated target code, exact quantizer objects/source snapshots, original job
commands/status and reports. Large model ELFs, checkpoint tensors/packed objects
and toolchains remain external; this is not a hermetic rebuild or admission
replay. Original runner commands are retained under each `job/spec.json`.
