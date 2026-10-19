# Complete TinyLlama cached sessions: reuse and implicit zero rows

Three complete executions of the same 128-token prefill plus 32 cached decode
steps preserve all **99 output/state hashes** and all **33 full logit tensors**.
The final variant uses **24.5646% fewer modeled execution cycles** than the
placement control, and **17.7607% fewer** than the already optimized variant.

| Measured work | Placement control | Reuse/connected | Plus implicit zero rows |
|---|---:|---:|---:|
| Reset | 4,505,788 | 4,505,788 | 4,505,788 |
| Prefill including preparation | 13,170,740,919 | 11,539,855,800 | 11,539,855,800 |
| 32 decodes including preparation | 60,890,057,577 | 56,393,309,899 | 44,327,119,790 |
| Complete measured execution | 74,065,304,284 | 67,937,671,487 | 55,871,481,378 |

The timing boundary sums reset, cross-program state preparation, and the
existing step compute intervals. Console printing and phase bookkeeping are
excluded in all three runs. Total preparation is unchanged at 9,012,365 cycles.
All actual simulator processes exited zero and printed the complete expected
132-event ABI protocol and `DONE`. These are modeled cycles, not simulator wall
time, and these numbers are not comparisons to the independent 160-token forward.

The placement control already includes prepacked immutable checkpoint operands,
unused-parameter pruning, and the same host placement of sensitive operations
and singleton embedding arithmetic. The second variant adds connected multiply
pairs, shared activation packing, and private BF16 matrix-to-VPU add transport.
It improves prefill by 12.3826%, decode by 7.3850%, and total execution by 8.2733%.

The final variant changes only the decode catalog object, encoding known zero
activation rows directly instead of materializing and quantizing padded F32
rows. It preserves the exact packed bytes, numerical refusal and partial-write
contract. Prefill and its stage objects are unchanged. Decode cycles fall by
**21.3965%** versus the second variant. One-token and prefill scheduling remain
distinct, derived from their actual logical/physical shapes. The independent
padding section evidence is retained in earlier task artifacts and tests;
this archive supplies the completed session measurement.

These frozen sessions precede the later connected RMS broadcasts, integer
word-group packing, guarded host math, movement-based tiling and reverse BF16
transport experiments. Do not attribute these cycle counts to those later
passes or infer their cached-session gains from independent forward timings.

Complete captured FP32 diagnostics are identical across all three runs: mean KL
0.07095571052, worst-row KL 0.1858635015, minimum cosine 0.9724467381, and top-token
agreement 28/33. All per-step metrics are retained. The capture uses fixed tokens;
these diagnostics do not establish generation quality or a new acceptance
threshold. There is no independent Nicolás cached-session result for this exact
capture here. The Nicolás-relative gate for the separate full forward is not
silently transferred to different session outputs. This session result establishes
preservation of the compiler control's complete outputs/state and quality.

| Whole-process counter | Control | Reuse/connected | Plus zero rows |
|---|---:|---:|---:|
| Commands | 10,106,912 | 9,797,152 | 9,797,152 |
| Native loops | 753,192 | 753,192 | 753,192 |
| Fences | 187,521 | 182,241 | 182,241 |
| VPU commands | 42,730 | 42,730 | 42,730 |
| Mesh tiles | 163,880,960 | 163,880,960 | 163,880,960 |
| Mesh busy cycles | 2,622,095,360 | 2,622,095,360 | 2,622,095,360 |
| Load bytes | 37,595,921,152 | 37,556,271,872 | 37,556,271,872 |
| Scale bytes | 1,155,792,896 | 1,155,792,896 | 1,155,792,896 |
| Store reads / Puts | 32,259,120 | 29,781,040 | 29,781,040 |

Counter scope covers the process and is wider than the summed execution timer.
Host tracking is off. Modeled execution instructions decrease from 68,960,539,297
to 62,836,031,628 to 50,772,517,196. The control-to-final cycle saving equals the
observed instruction saving plus the 5,800,805-cycle reduction in reported host
stalls. Cache/DRAM observations are retained, not presumed equal. The final
optimization saves host preparation without fewer native commands or bytes.

The linked ELFs are 2,429,826,080, 2,433,926,328 and 2,433,905,848 bytes. All use
the same 262,512,640-byte host operand pool, SHA-256
`fe29fc5c8f1281f5e512bc06c605d558a7ac5da32a218738e3fa38af8bfea04d`,
at `0x200000000`, and a 2 GiB arena at `0x110000000`. Each public KV state is
7,208,960 bytes and its associated length state is 176 bytes. Literal allocation
site sums and linked static extents are retained in receipts; they are not
peak-memory measurements.

```sh
python docs/evidence/cached_session_reuse_20261019/verify.py
# Also rehash the three retained external complete session images:
python docs/evidence/cached_session_reuse_20261019/verify.py \
  --build-root "$TMPDIR/mx_model_goal_20261018"
```

The standalone verifier requires Python and NumPy and imports no compiler code.
It independently reads the exact ordered console protocols, reconciles all
33 steps plus reset/preparation, checks 99 hashes and all 33×32,000 F32 logit
words per run, compares their bytes across variants, recomputes FP32 diagnostics,
and checks run/link/admission identities and native counter observations.

`executions.tar.xz` retains three full logs, counter reports, logit binaries,
original job commands/status and live process environments, link/stage receipts,
admission reports, generated callers, selected actual host/quantizer objects,
capture IR/inputs/FP32 references and checkpoint manifests. Every member is
pinned by `artifacts.json`. Large session images, device catalog objects, raw
checkpoint tensors and oversized generated input headers remain external;
their omission is explicit. This is not a hermetic rebuild or offline replay
of admission. Historical mutable quantizer sources are not reconstructed from
the current worktree.

To repeat the original execution, use each retained `job/spec.json` with its
unchanged externally retained link receipt, same captured session and toolchain
paths, and a fresh task-local output directory. Preserve all dump selections,
numerical placements, arena/memory, extension and timing settings. The archive
also retains the stage compilation recipes and generated session callers.
