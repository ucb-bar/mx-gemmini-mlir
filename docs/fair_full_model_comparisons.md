# Model and kernel comparisons

The completed results measure three workloads on the pinned Rocket-Spike
`e4m3_vpu` timing model with host tracking disabled. Only the 160-token
TinyLlama forward has an independent full-program performance baseline.
Cached TinyLlama and complete SmolVLA compare two compiler builds. There is
no accepted equal-quality Radiance speedup in the table below.
The newer [Radiance Rocket port](radiance_rocket_model_baseline.md) completes
the same full forward in 7,082,331,012 cycles. The fixed compiler release takes
45.81% more cycles and fails the full-160 mean-KL gate. A newer fused-gate configuration takes
5,849,962,853 cycles but fails the full/late top-token gates. The
[optimization measurements](evidence/radiance_rocket_optimization_20261019/README.md)
keep these unqualified timings and all completed alternatives separate from
the historical accepted comparisons below. The control's
[complete observations and profiles](evidence/radiance_rocket_forward_20261019/README.md)
remain available.

## Complete models

Speedup below is reference cycles divided by compiler cycles. Every row uses
all-FP8 offloaded matrices and the same inputs within its comparison.

| Workload | Reference | Reference cycles | Compiler cycles | Speedup | Cycle reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyLlama, all 22 layers + head, 160-token forward | Independent `npu-exploration` program, including embedding lookup | 12,762,360,525 | 10,326,818,367 | 1.236× | 19.08% |
| TinyLlama, 128-token prefill + 32 cached decodes | Corrected compiler baseline | 74,065,304,284 | 20,262,623,973 | 3.655× | 72.64% |
| SmolVLA, prefix + ten flow steps + action decode | Previous compiler build | 87,964,575,985 | 73,577,503,143 | 1.196× | 16.36% |

The baseline distinction matters: **3.655× is a compiler improvement, not a
speedup over independently stitched kernels**. The original 15,170,996,572-cycle
TinyLlama forward failed the numerical gate and is excluded from accepted
performance comparisons. The corrected forward baseline is 15,984,649,026
cycles; its complete output is bit-identical to the latest compiler result.

| Current session | Reset cycles | First program, including handoff | Recurrent program, including handoffs | Final action |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama cached | 4,505,788 | 8,882,396,857 prefill | 11,375,721,328 for 32 decodes | — |
| SmolVLA | 1,450,522 | 55,407,282,670 prefix | 18,168,761,303 for ten flow steps | 8,648 |

These totals include session reset, cross-program state preparation, and step
intervals. Console and phase bookkeeping are excluded. Historical step-only
session totals have a narrower boundary and must not be mixed with these
inclusive totals. The independent forward times token lookup through logits.

## Numerical comparison

TinyLlama's floating-point reference is the same checkpoint and input stream.
The independent source retains full distributions only at 33 aligned late
positions; the compiler retains all 160 rows. KL uses float64 log-softmax and
`KL(reference || quantized)` in nats. The source-relative gate permits at most
5% higher mean and worst-row KL and requires at least as many matching top
tokens on those same 33 rows.

| 160-token TinyLlama forward | Mean KL, 33 rows | Worst KL | Matching reference top tokens |
| --- | ---: | ---: | ---: |
| Independent quantized source | 0.0641713 | 0.1896031 | 27/33; 131/160 over all positions |
| Current compiler | 0.0637956 | 0.1699798 | 27/33; 133/160 over all positions |

The compiler passes this unchanged gate. All **5,120,000 F32 compiler output
words** are exact against the corrected compiler baseline. This establishes
numerical preservation for this input and source-relative gate; it is not a
perplexity or general model-quality evaluation.

Cached TinyLlama preserves all **99 output/state digests** and all **33 complete
logit tensors**, or 1,056,000 F32 words, across the measured transformations.
Its floating-reference diagnostic is mean KL 0.0709557, worst KL 0.1858635,
28/33 matching top tokens, and minimum logit cosine 0.972447. There is no
separate accepted cached-source quality threshold or independent complete
cached-source run.

SmolVLA preserves all **33 output/state digests** and **11 complete action
tensors**. Final action cosine is approximately **0.9752** against its original
floating-point trajectory, whose backbone retains native BF16 arithmetic.
F32 output files do not make that reference an entirely FP32 model. No
robot-task success threshold or dataset evaluation is established here.

## Why the compiler improves

TinyLlama's improvement combines corrected sensitive host placement, immutable
checkpoint prepacking, pruning unused entry weights, shared activation
packing, integer-group packing, connected BF16 matrix/VPU boundaries, VPU
broadcast/RMS preparation, guarded host math, and reuse of repeated attention
RHS preparation. The network graph exposes private temporary tensors and
repeated uses that a separately called kernel interface may conceal.

The latest cached step reduces **23,669,399,808 → 20,262,623,973** cycles
(14.39%) by supplying compact private RHS inputs instead of materializing
expanded repeated batches. Decode work falls **14,779,875,659 →
11,375,721,328** cycles. Device command counts and accelerator work counters
are unchanged; retired instructions fall from 18,581,902,169 to
15,175,110,570. This supports removal of host preparation work rather than
less model computation. The prefill objects are identical, but their cycle
count changes by 2,621,504 in the differently linked image; layout effects
prevent assigning the entire end-to-end difference to one source rewrite.

SmolVLA's latest comparison combines prepared VPU broadcast operands and
exact guarded host math with its existing packing/transport path. The output
checks cover the complete recurrent trajectory. Individual transformations
are documented in the linked evidence; their isolated percentage reductions
must not be added together.

Whole-network visibility enables these changes but does not guarantee a
faster result. Host packing, layout, numerical placement, and synchronization
still matter. A local kernel win is not automatically a model win. The
[completed transformation evidence](model_optimization_status_20261019.md)
records which changes reached full-model measurements.

## Source repositories and what was stitched

Two separate source repositories contribute controls:

- [`radiance-kernels`, `b6b9bf44af6d7ba88626b5d17b8b527390aec082`](https://github.com/ucb-bar/radiance-kernels/tree/b6b9bf44af6d7ba88626b5d17b8b527390aec082): Muon-issued MX GEMM/GEMV and GQA controls. These establish output and traffic properties, with no paired standalone Rocket-Spike cycle ratio.
- [`npu-exploration`, `0344bcba70b899039ccd0ac0801ca5605d46d3eb`](https://github.com/ucb-bar/npu-exploration/tree/0344bcba70b899039ccd0ac0801ca5605d46d3eb): the independent TinyLlama program, source matrix issuers, and paired MX+VPU attention heads used in timing comparisons.

The historical “stitched” full-session controls linked Merlin's captured
host graph, VPU objects, and session-state adapter with an alternative native
matrix-issuer catalog. Both sides kept that compiler graph and checked every
output digest. This measures **issuer substitution**, not an independently
written full-model schedule and not a full model assembled from Radiance
kernels.

| Historical issuer-only control | Compiler issuer cycles | Source issuer cycles | Compiler overhead |
| --- | ---: | ---: | ---: |
| TinyLlama, mixed FP4/FP8, prefill + 32 decodes | 83,547,564,353 | 83,544,602,795 | 0.00355% |
| SmolVLA, FP8, complete session | 180,219,637,915 | 179,451,403,096 | 0.42810% |

These older graphs, precision policies, and timing scopes differ from the
current builds. Dividing current totals by these old source-issuer totals
would not be a matched source-versus-compiler speedup.

The independent TinyLlama forward instead runs the full `llama_model.c`
program with the same token IDs and all 200 learned checkpoint tensors.
The adaptation supplies captured data/masks and adds the checkpoint embedding
lookup inside the timer: the original source accepted precomputed BF16
embeddings. All 327,680 embedding values are unchanged; including lookup
adds 58,646,741 source cycles. The archived source, patch, objects, log, and
[boundary receipt](evidence/nicolas_tinyllama_lookup_equal_boundary_20261015/README.md)
record the actual adaptation. Source and compiler may use different host/VPU
algorithms; the numerical gate is checked separately from equal logical work.

## Radiance kernel coverage

Each row binds generated packed operands/scales from the pinned source driver
and compares the compiler's BF16 output with that driver's reference.

| Kernel in `radiance-kernels` | M×N×K | Output comparison | Compiler weight traffic |
| --- | --- | --- | --- |
| `gemm_mxgemmini_ws` FP8 read-once | 256×64×2048 | 16,384 BF16 values exact | 32 tile reads; 131,072 bytes |
| `gemm_mxgemmini_ws_downproj_fp4` FP4 read-once | 256×64×5632 | 16,384 exact | 88 reads; 180,224 bytes |
| `gemm_mxgemmini_ws_restream` FP8 | 256×64×2048 | 16,384 exact | 128 reads; 524,288 bytes |
| `gemv_batched_fp8_m32`, `m64`, `m128` | 32/64/128×128×2048 | All BF16 outputs exact | One-read weight schedule checked |
| `gemv_batched_fp4_m128` | 128×128×2048 | All BF16 outputs exact | One-read weight schedule checked |

The first and third rows have different generated operands. Their 4× traffic
ratio is a schedule property, not an equal-input runtime speedup. All 16 GQA
P/V cutpoints are also checked, but source-produced probability tiles are
inputs to those controls. Muon softmax, online recurrence, finalization, and
mixed-engine launch are not covered as a complete attention kernel. One
source generator's `O_gold` also differs from executed source output; the
[qualification](evidence/radiance_kernel_comparison_20261014/README.md)
records that distinction.

Radiance's Muon+MX RTL timers and Rocket-Spike MX+VPU timers measure different
engines and boundaries. In particular, the published 42,364-cycle Radiance
flash-attention result and the 43,523-cycle standalone compiler control must
not form a speedup ratio. The audit reopens all seven measured driver files
and checks their hashes against the pinned receipt; it does not claim that
an unrelated compatible checkout is the entire historical repository.

## Paired timed attention heads

These source heads are from `npu-exploration`, not `radiance-kernels`.
Source and compiler run in the same guest with matched operands, arbitrary
Boolean masks, padded dimensions, timing boundary, and all 4,096 BF16 output
values exact per head. The compiler uses a checked cross-block QK/PV schedule.

| SmolVLA head-0 cutpoint | Source cycles | Compiler cycles | Speedup | Cycle reduction |
| --- | ---: | ---: | ---: | ---: |
| First-step, 113 keys | 16,717 | 16,033 | 1.043× | 4.09% |
| First-step, 163 keys | 22,057 | 21,382 | 1.032× | 3.06% |
| Final-step, 113 keys | 15,165 | 14,132 | 1.073× | 6.81% |
| Final-step, 163 keys | 16,956 | 15,733 | 1.078× | 7.21% |

These are separate head-level qualifications. Their schedule is not claimed
to be integrated into the current complete SmolVLA executable, and they do
not establish a whole-attention or full-model speedup.

## Reproduce and audit

Follow the [model guide](compile_e2e_models.md) for capture identities, exact
policies, stage compilation, session linking, execution, complete output
checks, and FireSim packaging. Run the portable retained-artifact audit with:

```sh
python -m tools.audit_published_models --out /path/to/new-audit
```

Add `--build-root /path/to/retained-builds` to reopen the original large
executables and objects, and `--radiance-root /path/to/radiance-kernels` to
reopen source drivers. Without the build root, the audit checks retained
archives and their hash bindings rather than unavailable executable bytes.
The [audit report](model_results_audit.md) describes exactly what was reopened.
