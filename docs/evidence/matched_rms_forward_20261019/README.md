# Complete TinyLlama forward: connected broadcast multiplies

The connected-broadcast pass reduces the independent 160-token, 22-layer
token-to-logits forward from **13,735,943,708 to 13,479,416,598 modeled cycles**:
**1.8676%** below its matched control. All **5,120,000 FP32 output words** are
bit-identical, SHA-256
`d62f10e03e9587fb4e77cba002f5973368075b51bf17628926a6e5459ac6ff2a`.
Both actual simulator processes completed successfully, with complete dumps,
accepted structural admissions and uninstrumented stage timers.

The candidate is 15.6727% below the corrected quality-preserving compiler
baseline of 15,984,649,026 cycles. It remains **5.6185% slower** than Nicolás's
matched token-lookup-to-logits control, 12,762,360,525 cycles. This does not
establish a win over that source implementation. The older 15.171B compiler
result has different numerical quality and is not an equal-quality control.

The unchanged Nicolás-relative gate passes on the **33 retained full source
logit distributions**: compiler mean KL 0.06379562704, worst-row KL
0.16997975795, and FP32 top-token agreement 27/33. Source values are
0.06417128458, 0.18960313411 and 27/33. The original 5% KL margin and top-token
requirement remain unchanged. Compiler FP32 top-token agreement across all 160
positions is 133/160. Complete source distributions for all 160 positions were
not archived; the full compiler dump does not supply missing source rows.

Merlin's connected-call legality handles broadcast input shapes and dimensions.
MX implements the two-stage multiply on private VPU scratch storage, retaining
the required intermediate BF16 rounding. This full candidate connects 45
broadcast multiply pairs. The explicit selection is
`--connect-vpu-broadcast-multiplies`. This comparison does **not** yet include
`--prepare-vpu-broadcast-operands`, integer group packing, guarded host math or
the experimental native tile selector; their separate full runs were pending
when this archive was created. The isolated section work and refusal cases are
retained in [broadcast_multiply_20261019](../broadcast_multiply_20261019/README.md).

| Whole-process counter | Control | Connected |
|---|---:|---:|
| Commands | 3,542,604 | 3,312,204 |
| Native loops | 38,608 | 38,608 |
| Fences | 495,601 | 492,001 |
| VPU commands | 29,364 | 29,364 |
| Load bytes | 4,227,602,752 | 4,198,111,552 |
| Scale bytes | 115,279,872 | 115,279,872 |
| Mesh tiles | 40,970,240 | 40,970,240 |
| Mesh busy cycles | 655,523,840 | 655,523,840 |

These counters cover a wider process boundary than the compute timer; host
tracking is off. The linked ELF shrinks from 1,330,824,624 to 1,330,818,608 bytes.
The sum of literal allocation-site sizes decreases from 1,431,470,834 bytes at
1,106 sites to 1,372,484,654 bytes at 1,061 sites. These are allocation-site sums,
not peak-memory measurements. Both have zero dynamic allocation sites and use
the original 2 GiB arena and admitted memory layout.

Both full builds used the same checkpoint operands and pinned quantizer
revision. The exact `runtime_quant.o` is retained for each and is byte-identical
(14,496 bytes, SHA-256
`f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0`).
The historical quantizer C/header/LUT snapshot is retained under `frozen_quant`;
it is distinct from the later generic word-packing revision. `runtime_compiler`
receipts describe the scalar MLIR support object, not `runtime_quant.o`.
The control reproduces the previous connected-sigmoid full result's cycles and
output words; this archive retains the newly built control itself.

```sh
python docs/evidence/matched_rms_forward_20261019/verify.py
```

The verifier uses Python, NumPy, the digest-pinned adjacent full-output reader,
and the existing independent-source archives. It imports no compiler package.
It reopens both full logs, checks every compiler word and both runtime hashes,
checks receipt/admission/recipe identities, reconstructs the source-relative
quality gate and checks the token-to-logits source timer. Optional verification
of the actual external linked images is available:

```sh
python docs/evidence/matched_rms_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The compact archive retains full logs, capture IR/input identities, build and
terminal validation receipts, generated target sources, transformed IR and
rewrite records, original job specifications/status, source snapshots and the
actual quantizer objects. `artifacts.json` pins every member and records its
original path. Large model ELFs, packed checkpoint arrays/objects and transitive
toolchains remain external. Receipt verification is not an offline replay of
ELF admission or a hermetic rebuild.

Original complete commands are `baseline/job/spec.json` and
`candidate/job/spec.json` in the archive. Reproduce in fresh task-local workdirs
with the same capture, target profile, execution policy, checkpoint reuse,
extension, `GEMMINI_MODE=both`, `GEMMINI_PERF_CONFIG=e4m3_vpu`, and
`GEMMINI_PERF_SET=mem.host_tracking=0`. Preserve the original timing boundary and
complete 5,120,000-word dump selection. Then run
`tools.validate_completed_merlin_stage` followed by
`tools.compare_matched_tinyllama_forward` with the retained same-input source
archive and `nicolas_tinyllama_lookup_equal_boundary_20261015` boundary archive.
