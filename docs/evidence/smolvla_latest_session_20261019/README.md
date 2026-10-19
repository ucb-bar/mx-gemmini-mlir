# Complete SmolVLA session: connected preparation and host execution

The optimized compiler completes prefix encoding, ten flow-denoising steps and
action decoding in **16.3555% fewer modeled cycles** than its fresh matched
compiler control. All 33 output/state hashes and eleven complete 6,400-byte action
tensors match exactly. Both simulator processes completed and exited zero.

| Measured work | Control cycles | Optimized cycles |
|---|---:|---:|
| Reset | 1,450,522 | 1,450,522 |
| Prefix, including preparation | 66,127,433,386 | 55,407,282,670 |
| Ten flow steps, including preparation | 21,835,683,429 | 18,168,761,303 |
| Action decode, including preparation | 8,648 | 8,648 |
| Complete measured execution | 87,964,575,985 | 73,577,503,143 |

The boundary sums reset, program-state preparation and the existing step compute
intervals. Console printing and phase bookkeeping are excluded in both runs.
Preparation is unchanged: 31 prefix cycles, 1,448,533 across flow steps and 4,398
for action decoding. Measured instructions fall from 86,034,129,849 to
71,650,743,356. These are modeled cycles, not simulator wall time.

Both builds already use immutable checkpoint preparation, unused weight pruning,
shared activation packing and equivalent BF16 read-view reuse. The candidate adds
generic word-group preparation and fixed-axis inlining, connected broadcast
preparation, and exact guarded host math. Its prefix selects 25 connected
multiply pairs; flow selects none. Neither stage selects a connected sigmoid
chain or a matrix/VPU transport edge. Movement-based tile selection, virtual-zero
decode padding, repeated batch RHS reuse and compact RHS inputs are not selected
in this pair. This measures the combined applicable bundle; it does not attribute
the entire gain to any single pass.

Only prefix `model.o`, prefix `device_catalog/mx_catalog.o` and flow
`device_catalog/mx_catalog.o` change among linked stage objects. Action and stage
adapter objects are unchanged. The fresh control is 10,070 cycles above the
[earlier widening-reuse candidate](../smolvla_widening_session_20261019/README.md);
that historical image is not substituted for this experiment's control.

| Whole-process observation | Control | Optimized |
|---|---:|---:|
| Commands | 18,034,641 | 17,727,441 |
| Loops | 446,775 | 446,775 |
| Fences | 201,109 | 196,309 |
| Mesh tiles | 48,141,352 | 48,141,352 |
| Mesh busy cycles | 770,261,632 | 770,261,632 |
| VPU commands | 142,409 | 142,409 |
| Device load bytes | 12,580,427,280 | 12,541,105,680 |
| Load Gets | 340,109,841 | 337,652,241 |
| Store reads | 81,102,340 | 78,644,740 |
| Store Puts | 89,721,028 | 87,263,428 |
| Scale bytes | 315,957,264 | 315,957,264 |

These counters have a wider scope than the summed execution intervals. Host
tracking is off. Store transaction counts are not converted into byte savings.
The reduction includes 307,200 commands, 4,800 fences and 39,321,600 load bytes,
with unchanged mesh and VPU arithmetic work.

There is a material memory tradeoff. The final ELF grows from 485,475,216 to
493,841,984 bytes, an increase of 8,366,768 bytes. Allocated BSS sections sum to
18,091,140 and 26,908,292 bytes, respectively. ELF growth must not be interpreted
as code growth alone. Prefix literal allocation sizes sum to 4,398,657,469 and
4,320,012,169 bytes across 2,204 and 2,179 sites, respectively; both retain one
dynamic-sized site. Flow is unchanged at 184,877,226 bytes across 1,308 literal
sites and no dynamic-sized site. These sums are not peak memory. Both sessions
use a 6 GiB arena at `0x210000000`, the same 97,736,960-byte host checkpoint pool
at `0x200000000`, and the same admitted simulator memory extent. Full section
tables, image hashes, object hashes and allocation observations are retained in
`resources.json` inside the archive.

The FP32 diagnostic is unchanged: final cosine 0.9752036640, MAE 0.03336054171,
RMSE 0.07782276601, normalized RMSE 0.2398155381 and maximum absolute error
0.4540897161. All ten flow-step diagnostics are retained and recomputed by the
reader. No independent complete Nicolás SmolVLA baseline or action-quality
acceptance threshold has been established. This result preserves the compiler
control's complete outputs and numerical behavior.

Standalone verification requires Python and NumPy, with no compiler imports:

```sh
python docs/evidence/smolvla_latest_session_20261019/verify.py
```

The reader checks the ordered twelve-step protocol and all 45 events, reset and
preparation boundaries, all hashes and full tensors, FP32 diagnostics, device
counters, unchanged inputs and policies, selected transformations, link
identities and retained admission reports. The guarded math check binds the
actual wrapper, object, original `libm.a`, selected symbol suppliers and final
image recipe. The frozen linker producer is recovered from its recorded source
revision and checked against the original recipe; current mutable source is not
substituted for the historical producer.

To additionally rehash both actual retained images and independently read their
allocated ELF section tables:

```sh
python docs/evidence/smolvla_latest_session_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

`executions.tar.xz` contains 3,338 unique file bodies representing 5,559 logical
files in 24,117,220 compressed bytes. `artifacts.json` pins every logical file's
size, SHA-256 and original path; its alias map restores duplicate bodies in
memory during verification. It retains full runtime logs, output binaries,
counter reports, process environments, commands/status, generated sources and
headers, capture inputs and FP32 references, compiler/link receipts, applicable
small objects, policies, and the selected host math library. Large stage objects,
linked ELFs, raw checkpoint payloads and other toolchain dependencies remain
external. This is compact measurement evidence, not a hermetic rebuild or an
offline replay of ELF admission.

For reproduction with the original images, read each retained
`reference/job/spec.json` and `candidate/job/spec.json` and replay its runner
command with a fresh task-local output directory. Both use the original
`smolvla_latest_{control,optimized}_link/mx_session_link.json` and original
captured session. Stage compilation and final link recipes are retained for
rebuilding with the recorded inputs and tools. Preserve checkpoint bytes,
profile, simulator settings, numerical placement, dump selection and timing
boundaries. A rebuild with newer compiler sources is a new experiment.
