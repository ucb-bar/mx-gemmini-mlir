# Guarded compact RHS inputs in cached TinyLlama

The complete 128-token prefill plus 32 cached decodes finishes successfully
with **20,262,623,973 inclusive modeled cycles**, versus **23,669,399,808**
for ordinary repeated-RHS packing reuse: **14.3932% fewer cycles**. Against
the original corrected cached compiler's 74,065,304,284 cycles, this is
3.655× faster. Neither reference is an independent source application.

| Inclusive interval | Ordinary RHS | Guarded compact RHS |
| --- | ---: | ---: |
| Reset | 4,505,788 | 4,505,788 |
| Prefill and preparation | 8,885,018,361 | 8,882,396,857 |
| 32 decodes and preparations | 14,779,875,659 | 11,375,721,328 |
| Total | 23,669,399,808 | 20,262,623,973 |

Every one of the 99 output/state digests matches. All 33 complete logit
tensors, including the console dump bytes, match exactly: 1,056,000 F32
words. The floating-point diagnostic is unchanged: mean KL 0.0709557105,
worst KL 0.1858635015, 28/33 top-token matches, minimum cosine 0.9724467381.
A separate cached-source numerical acceptance threshold is not established.

The decoder admits 44 private compact-RHS uses with the cost/privacy guard.
Only its model object and MX catalog change. The prefill objects are identical,
but their measured time differs in the final linked image; this comparison
includes that layout effect. Retired instructions fall from 18,581,902,169 to
15,175,110,570. Device commands, loops, fences, mesh tiles, busy cycles,
VPU work, and device transfer counters are unchanged. The main saving removes
host expanded-input materialization rather than model computation.

Literal allocation sizes sum to 413,326,750 → 357,947,128 bytes across
1,087 → 1,064 sites. These are not peak-memory measurements. The 2 GiB arena
and native workspace are unchanged. ELF size decreases from 2,439,205,960
to 2,439,169,264 bytes. The candidate SHA-256 is
`1750d71086f9aec925df1e27fd07019cbd4e86abd5487bc1004232b42ed1e161`.

The archive retains 970 logical members, with 969 stored bodies in 4,214,724
compressed bytes. It composes its reference from the hash-pinned
[cached RHS archive](../cached_rhs_session_20261019/README.md), preserving
the original capture and earlier baseline chain. It retains generated source,
IR, catalogs, small objects, commands, receipts, complete output streams,
quality diagnostics, counters, and resource observations. Large checkpoint
files and executable images remain external, bound by size and SHA-256.
The linker producer sources for both sides are frozen separately. Checkpoint
object storage reuse was permitted only after independent production and a
complete byte comparison; no runtime weights or computation were omitted.

Reopen all portable evidence with Python and NumPy:

```sh
python docs/evidence/cached_compact_rhs_session_20261019/verify.py
```

To reopen the actual retained executables, section tables, and stage objects:

```sh
python docs/evidence/cached_compact_rhs_session_20261019/verify.py \
  --build-root /path/to/retained-builds
```

Both checks passed for the published result. Timing uses the pinned
`e4m3_vpu` Rocket-Spike model with `mem.host_tracking=0`, including reset and
program-state handoff but excluding console/phase bookkeeping. No FPGA
measurement is claimed. See the [comparison guide](../../fair_full_model_comparisons.md)
for the independent-forward and source-kernel qualifications.
