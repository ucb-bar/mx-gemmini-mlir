# TinyLlama and SmolVLA MX+VPU coverage against Nicolas's kernels

**Conclusion: neither model routes every plausible MX+VPU operation to the device.**
The existing catalog's `coverage_complete` means every *selected* contraction
has a compiled binding. It says nothing about unselected contractions, VPU
regions, bias epilogues, or fusion across operations.

This audit uses the full TinyLlama prefill/decode captures, the xDSL 0.65
SmolVLA prefix/flow/action captures, the selected
`MxE4M3Fp4VpuGemminiRocketConfig` profile (SHA-256
`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`),
and Nicolas's `npu-exploration` source at commit
`0344bcba70b899039ccd0ac0801ca5605d46d3eb`. TinyLlama's prefill source
SHA-256 is `8b7e6e709fd12352773b5a7b264916b3f98b9a6b901f329de04446619ef09031`;
SmolVLA prefix source is `f8f1f15d1aef8adcc630042e9c808880b02129b082a329ccd8b02ec93ecba285`;
flow source is `2e709838aee113a2678d838dc4389943438e6487793bb3b426b3707868724e17`.

| Captured stage | Contraction regions | MX catalog bindings | Remaining host regions |
| --- | ---: | ---: | --- |
| TinyLlama prefill and decode, each | 200 | 199: 155 FP4 matmuls, 44 FP8 attention batch matmuls | One rotary-embedding batch matmul with K=1 |
| SmolVLA prefix | 316 | 211: 186 FP4, 25 FP8 | 32 vision-attention batch matmuls; 73 bias-add regions paired with already-routed `addmm` matrix halves |
| SmolVLA flow | 152 | 116 FP4 | 32 attention batch matmuls; four bias-add regions paired with already-routed `addmm` matrix halves |
| SmolVLA action decode | 0 | 0 | No contraction proposal |

The **64 SmolVLA attention batch matmuls are a concrete matrix coverage gap**.
All are f32 rank-three source contractions, and all 64 pass Merlin's
zero-initialized f32 catalog source matcher when the MX target manifest is
loaded. The target's checked FP8 native issuer also constructs legal padded
schedules for all six distinct shapes: vision QK/PV
`(15,113,113,64)` / `(15,113,64,113)` and flow QK/PV
`(15,50,163,64)` / `(15,50,64,163)` and
`(15,50,113,64)` / `(15,50,64,113)`. This is shape and binding evidence,
not a Spike numerical qualification for those newly selected sites. The
current precision planner recognizes QK/PV only when two batch matmuls carry
an attention module name; these 64 have no captured `fqn` and are classified
as `host`. It must recover the attention relation from graph dataflow or
repair provenance, then qualify the resulting FP8 path.

The 77 host `addmm` regions above are **bias epilogues**, each paired by
module name with an FP4 matrix component that already has an MX binding.
They are not 77 additional missed matrix multiplies. A resident bias-add
epilogue could avoid a host round trip, but the current catalog returns the
matrix result to host before applying bias. The lone TinyLlama K=1 batch
matmul is in `rotary_emb`; its MX padding is shape-admissible, but assigning
it FP8 would change the captured arithmetic and has no accuracy or cost
qualification. It remains an explicit host choice, not proven device coverage.

The VPU gap is separate. The captures have 22 TinyLlama, 28 SmolVLA prefix,
and 16 SmolVLA flow softmax regions, all host-owned. Their score/probability
boundaries are f32 rank four, while the current declared direct VPU contract
is BF16 rank-two row softmax. Nicolas's `attn_vpu.c` and `attn_flash.c` show
QK, VPU softmax, requantization, and PV on chip; `llama_layer_e2e.c` also
uses VPU primitives for RMSNorm, RoPE, SwiGLU, and residual arithmetic.
Those source kernels demonstrate hardware patterns, but the compiler has no
model-wide legalization, numeric policy, or generated schedule for these
captured regions. The current profile contract declares only MX matmul and
direct BF16 row softmax; it does not certify the other VPU sequences.

Even the bound matrix operations are dispatched individually. Nicolas's
resident QKV, attention, and MLP chains keep intermediate tiles on chip; the
current full-model catalog materializes a host boundary between contractions.
Native `mk` loop scheduling now narrows the per-contraction command gap but
does not create those graph-level fused schedules.

Audit method: `build_mx_worklist_from_merlin` and
`build_model_precision_plan` counted every captured contraction region;
the catalog manifests supplied exact bindings; `catalog_source_contractions`
checked the 64 omitted f32 batch matmuls; `emit_dense_batch_contraction`
checked the six distinct padded FP8 schedules; and
`audit_softmax_boundaries` supplied the VPU ownership counts. No benchmark
name or source-region ID was used to choose a device route.
