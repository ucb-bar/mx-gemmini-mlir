# Reverse BF16 transport in a complete TinyLlama forward

The connected VPU producer now returns its private BF16 representation directly
to shared matrix activation preparation, avoiding a temporary F32 widening and
renarrowing. Complete graph/privacy, layout, capacity and synchronous-completion
checks still apply. The compiler selects 44 private edges. Both runs use the
same current BF16-input-capable helper source and movement-bounded native tiles;
only the candidate selects `--vpu-matrix-transport`.

| Completed token-ID-to-logits execution | Modeled cycles |
|---|---:|
| Fresh compiler control | 11,094,643,214 |
| Reverse BF16 transport | 11,008,973,639 |
| Retained independent Nicolás, same logical boundary | 12,762,360,525 |

The matched compiler reduction is **0.77217%**. The candidate is **13.7387%
below Nicolás's cycle count**, and 31.1278% below the corrected compiler's
15,984,649,026 cycles. The older 15,170,996,572-cycle compiler archive has
different numerical quality and is not this control. The fresh control differs
by 3,520 cycles from the earlier native-tile archive because its helper source
has changed; the isolated comparison uses this fresh matched pair.

Both complete 160-token, 22-layer executions preserve every one of the
**5,120,000 F32 logit words**, SHA256
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
The unchanged source-relative 5% gate passes on 33 archived complete source
distributions: mean KL 0.0637956270, worst-row KL 0.1699797580, top agreement
27/33. Compiler agreement with the FP32 top token is 133/160. Independent full
source distributions exist for only 33 positions. Gate acceptance is not a
claim of source bit identity.

Wider process counters match: 3,312,204 commands, 38,608 loops, 492,001 fences,
29,364 VPU commands, 40,970,240 mesh tiles, 655,523,840 mesh busy cycles,
3,939,506,496 load bytes and 107,198,464 scale bytes. Host tracking is off.
This change saves host representation work, without a measured reduction in
these device transfer/work counters. Inputs, checkpoint tensors, profile,
simulator extension, `both/e4m3_vpu/mem.host_tracking=0` settings and logical
uninstrumented timing boundary match.

Images are 1,330,818,680 → 1,330,837,896 bytes. Literal allocation sizes sum to
1,372,484,654 → 1,343,648,814 bytes across 1,061 sites, with no dynamic-sized
sites. That 28,835,840-byte reduction is not a measured peak-memory reduction.
Both retain the 2 GiB arena at `0xd0000000`. Frozen quantizer C SHA is
`f58d83932541d540a2a39f379280a8f2dbeca93059604aa7d9c436db66d521b1`.
The actual quantizer objects differ because the candidate includes the selected
BF16-input endpoint; each original object and its compilation record is pinned.
The earlier native-tile archive uses a different frozen source/object.

```sh
python docs/evidence/matched_reverse_transport_forward_20261019/verify.py
# Also rehash both original complete images while available:
python docs/evidence/matched_reverse_transport_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone stdlib/NumPy verifier reopens 1,762 retained files, complete logs,
all full logit words, original source-relative gates, counters, recipes,
admissions, live environments, native tile identities, frozen source/objects,
guarded host math and all 44 recorded private edges. Both actual 1.33 GB image
rehashes also passed. The 19,409,000-byte compact archive retains sources,
generated IR, source coverage, exact commands and completed job observations.
Large images/checkpoint tensors and toolchain executables remain external;
this is not a hermetic rebuild or independent instruction-level proof.

Reproduce from `baseline/job/spec.json` and `candidate/job/spec.json` command
arrays in fresh task-local directories. Both use the original full capture,
explicit numerical policy, all listed packing/pointwise/guard options,
`--native-tile-policy movement`, structural admission and output cap5,120,000.
Add `--vpu-matrix-transport` only to the candidate. Validate each completed
output with `tools.validate_completed_merlin_stage`, then compare with
`tools.compare_matched_tinyllama_forward` against the original independent
source and token-lookup boundary archives. The native endpoint and refusal
qualifiers remain in [vpu_matrix_transport_20261019](../vpu_matrix_transport_20261019/README.md).
