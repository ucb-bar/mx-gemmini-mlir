# Merlin schedule IR to MX native loops

The [receipt](receipt.json) records three model-derived kernels executed on the reference implementation’s pinned
Gemmini MX Spike extension. The two FP4 MLP projections use captured small-LLaMA f32 operands,
the target-owned runtime quantizer, and the existing three-pointer Merlin catalog ABI. The FP8
attention QK head uses captured packed TinyLlama operands. Each output matched its archived
independent BF16 golden exactly.

For each case, `schedule.mk` is Merlin's canonical `mk` loop IR. New whole-model catalogs also
include a verified `.schedule.mlir` plan per kernel. The MX builder reads its selected policy
before constructing the `mk` schedule. The MX instruction set checks the schedule's
buffer bounds, alternating scratchpad slots, direct precision, and target profile; Merlin's generic
C emitter renders the `mx_issue.c` in the same directory. The schedule has counted M, N, and K
loops, so it also represents M=1024 SmolVLA rows without copying a handwritten C loop.

The FP4 gate schedule issues 94 host RoCC commands versus 677 in the source-qualified serial
schedule; down issues 94 versus 802. A separate full TinyLlama prefill catalog compiled 199 exact
source bindings to seven `mk` scheduled RV64 kernels. Its estimated dynamic host RoCC issue count
is 129,226, versus 29,218,058 for the serial catalog. These are **command counts**, not measured
cycle counts or speedups: the native hardware loop performs transfers and compute internally, and
runtime f32 quantization plus host graph operations still cost time.

The [linked full-checkpoint TinyLlama native prefill](tiny_native_prefill_spike.log) reached guest
`DONE` on the reference implementation’s pinned Spike with zero memref rank mismatches. Its 32,000 f32 logits have the
same SHA-256 as the [serial MX prefill](../tiny_full_checkpoint_prefill_spike_20261011/README.md):
`e829ba982b701c5c8e2b64af8d9279cd23a106ddfa84ca147d6c82a00f4b6459`.
The native run reported 374,780,809,774 simulated cycles versus 374,944,319,266 for the serial
run, a 0.044% reduction. This single stage result does not establish a meaningful speedup. The
ELF was manually relinked with a 5 GiB arena; the detached launcher did not retain a process exit
code. Its native catalog object is byte-identical to the final catalog with `schedule.plan` and
`mk` artifacts.

The same profile and schedule path also compiled the selected contraction catalogs for Qwen
prefill (253 bindings, seven kernels), Gemma prefill (235, eight), SmolVLA prefix (211, twelve),
and SmolVLA flow (116, ten). Their object and source digests are in the receipt. This establishes
per-contraction code generation and linking readiness; it does not establish whole-model execution
or numeric parity for those stages.

The [M=1024 row-tile probe](fp4_m1024_serial_parity/spike.log) ran the serial and native issuers
on identical nonzero encoded FP4 inputs. All 32,768 BF16 results agreed. This is schedule parity,
not an independent arithmetic golden.

The general `schedule` dialect records placement, packing, and dispatch policy at the graph level.
`mk` is its per-kernel loop schedule IR; this MX backend supplies the native instruction semantics.
The existing model stage still has per-contraction host boundaries. Resident QKV and G/U chains,
VPU softmax placement, and full-model numerical qualification remain separate compiler work.

To regenerate either FP4 site with a checkout matching the pinned profile, run
`python -m tools.qualify_runtime_f32_projection_spike --site g --dense-adapter
--native-scheduled-issuer --rtl-root <pinned-gemmini> --riscv-root <riscv-tools>
--out-dir <new-output>` (replace `g` with `dn` for the down projection).
