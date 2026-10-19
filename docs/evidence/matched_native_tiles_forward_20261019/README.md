# Movement-bounded tiles in the complete TinyLlama forward

The explicit native tile policy reduces complete token-ID-to-logits execution
from **11,105,781,721 to 11,094,646,734 modeled cycles**, a **0.100263%** reduction.
Both runs execute the same captured 160-token, 22-layer forward and preserve
all **5,120,000 F32 logit words** exactly. The candidate uses **13.0674% fewer
cycles** than the reference implementation’s retained equal-boundary 12,762,360,525-cycle execution
and **30.5919% fewer** than the corrected compiler's 15,984,649,026 cycles.
The older compiler archive with different quality is not this control.

The unchanged source-relative gate passes: mean KL 0.0637956270, worst-row KL
0.1699797580 and 27/33 top-token agreement on the 33 retained full source
distributions. The compiler's full 160-row output is retained and checked;
independent source distributions are available for only 33 rows. This is the
existing 5% gate, not a new numerical threshold or a claim of source bit identity.

| Wider process counter | Legacy tiles | Movement-bounded tiles |
|---|---:|---:|
| Commands | 3,312,204 | 3,312,204 |
| Matrix loops | 38,608 | 38,608 |
| Fences | 492,001 | 492,001 |
| VPU commands | 29,364 | 29,364 |
| Mesh tiles | 40,970,240 | 40,970,240 |
| Mesh busy cycles | 655,523,840 | 655,523,840 |
| Load bytes | 4,198,111,552 | 3,939,506,496 |
| Scale bytes | 115,279,872 | 107,198,464 |

Load bytes fall 6.1600%; complete execution improves much less because the
timer includes host work and other operations. Host tracking is off, and these
process counters cover a wider boundary than the model timer. Both runs use
the same pinned `both/e4m3_vpu/mem.host_tracking=0` simulation, checkpoint,
profile, inputs, guarded host math and uninstrumented logical timing boundary.

Both images are 1,330,818,680 bytes, with a 2 GiB arena at `0xd0000000`.
Literal allocation sites sum to 1,372,484,654 bytes across 1,061 sites and zero
dynamic-sized sites. This sum is not peak memory. Frozen quantizer C SHA256 is
`a6e73983348ed9a8d1348cc6c656cd4cb813ef378c7c1b7a5bc97f486ce29c2e`
and the actual selected quantizer object is identical in both images. The later
BF16-input helper refactor is separately qualified and is not substituted for
these compiled sources.

The generic bounded search lives in Merlin. MX supplies native layout,
accumulator and transfer facts. The fixed-K candidate must not increase any
selected integer work/resource bound. Defaults retain legacy selection; FP4
retains its original route. Cold and decode regressions and accepted native
cases remain in [native_tile_search_20261019](../native_tile_search_20261019/README.md).
This result supports the explicit measured FP8 candidate; it does not establish
universal profitability of larger tiles or asynchronous overlap.

```sh
python docs/evidence/matched_native_tiles_forward_20261019/verify.py
# Also rehash both original complete images while they remain available:
python docs/evidence/matched_native_tiles_forward_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone verifier uses stdlib and NumPy, imports the pinned adjacent
retained forward reader without compiler imports, reopens both complete logs
and all logit words, recomputes the original source gate, and binds counters,
captures, policies, admissions, recipes, frozen quantizers and guarded-math
objects. The compact archive contains 1,455 files (19,362,548 compressed bytes).
Large images, checkpoint tensors and toolchain executables remain external;
optional image rehashing checks custody without claiming a hermetic rebuild or
independent instruction qualification.

The archived `baseline/job/spec.json` and `candidate/job/spec.json` retain exact
command arrays and explicit environment selections. Reproduce in fresh task-local
directories using `tools.build_merlin_stage`, the same capture/policy/checkpoint,
all listed preparation/connection/guard options, full output cap 5,120,000 and
structural admission. Select `--native-tile-policy movement` only for the candidate.
Validate each completed output with `tools.validate_completed_merlin_stage`, then
run `tools.compare_matched_tinyllama_forward` against the retained independent
source and token-lookup boundary archives. Timing and quality are required from
completed full executions; native section measurements alone are insufficient.
