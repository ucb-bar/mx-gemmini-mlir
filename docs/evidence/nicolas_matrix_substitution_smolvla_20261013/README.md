# SmolVLA MX matrix issuer comparison

The comparison uses the captured SmolVLA flow stage, its checkpoint-backed
weights, the same host program, and the same compiled VPU kernels. The only
replacement is each MX matrix issuer: the alternate object calls the native
loops in Nicolas's `baremetal/mxgemmini/include/mx_native.h` at
`npu-exploration` commit `0344bcba70b899039ccd0ac0801ca5605d46d3eb`.
The source header's SHA-256 is
`98bc0384dbf1917af9427dd30a51770063e56152933259d5fb32e5e04a5505e8`.

Both runs use Nicolas's MX+VPU Spike extension, `GEMMINI_MODE=both`, the
`e4m3_vpu` timing preset, and `mem.host_tracking=0`. The replacement uses
128-row native source tiles. These are modeled Spike cycles, including host
execution. They are not the standalone RTL mesh cycles in Nicolas's
`README_PERFORMANCE.md`.

| One flow step | Compiler issuer | Nicolas native issuer |
| --- | ---: | ---: |
| Modeled cycles | 2,112,653,481 | 2,112,919,925 |
| MX commands | 555,735 | 555,139 |
| Mesh busy cycles | 7,676,544 | 7,676,544 |
| VPU commands | 4,252 | 4,252 |

The native issuer takes 266,444 more cycles, or 0.0126%. All three declared
output SHA-256 digests match exactly. The [compiler log](flow_compiler_spike.log),
[native log](flow_nicolas_spike.log), [compiler counters](flow_compiler_perf.txt),
and [native counters](flow_nicolas_perf.txt) preserve the measurements.
The [replacement source](flow_issuers.c) and [shape receipt](flow_substitution.json)
record the generated source calls. An earlier exploratory FP4 row-pitch
mistake was corrected before this measurement; the packed activation
row-pair pitch is physical `K` bytes.

The [pruned flow operation profile](../smolvla_flow_fixed_rhs_20261013/README.md)
attributes 1,208,877,122 cycles to whole MX adapter calls, 501,696,666 to
VPU calls, and 385,950,608 to other host operations. MX adapter intervals
include live operand conversion and readout, so their total is not mesh
execution time. The unchanged 7,676,544 mesh busy cycles and near-equal
stage timing show that changing only the matrix issuer has little effect on
this model step.

The same replacement was built for prefix encoding, which has 14 distinct
issuer shapes and larger row counts. The [prefix receipt](prefix_substitution.json)
records its object hashes. The [full-session identity record](full_link_identity.json)
shows that only the prefix and flow catalog objects changed in the linked
12-step session; host objects, captured programs, weights, profile, and memory
layout match. A full-session cycle comparison requires both completed Spike
runs and matching output digests. The
[`compare_linked_session_cycles` tool](../../../tools/compare_linked_session_cycles.py)
enforces those checks before reporting a ratio.

This experiment isolates matrix issuance. It does not replace the host graph
with Nicolas's resident QKV, attention, or MLP chains. His existing full
TinyLlama layer computes RMSNorm, RoPE, softmax, SwiGLU, and residuals on the
host. The separate VPU attention kernels use other fixed shapes and data
layouts. A whole-model handwritten MX+VPU comparison needs those chains
adapted to the exact captured model, precision policy, masking, and output
contract before its cycle count is comparable.

The [source-bound attention inventory](../smol_ssa_attention_coverage_20261011/README.md)
identifies the concrete integration work:

| SmolVLA stage | QK shape `(batch, query, key, depth)` | Pairs | Source adaptation |
| --- | --- | ---: | --- |
| Prefix vision | `(12, 1024, 1024, 64)` | 1 | Query tiling and streamed keys |
| Prefix 113-key | `(15, 113, 113, 64)` | 16 | Pad to 128×128 and mask keys |
| Flow 163-key | `(15, 50, 163, 64)` | 8 | Pad to 64×192 and mask keys |
| Flow 113-key | `(15, 50, 113, 64)` | 8 | Pad to 64×128 and mask keys |

These padded shapes describe possible schedules, not numerically qualified
replacements. Fusion may change intermediate rounding, so it needs an
accuracy gate against the same captured model before comparing whole-model
cycles.

A separate [equal-work dense attention comparison](../attn_vpu_equal_work_20261013/README.md)
measured 43,641 source versus 43,523 compiler cycles for one 64×256×128
MX+VPU kernel. That paired kernel result cannot be multiplied into a
SmolVLA whole-model cycle estimate: its shapes, host boundaries, and
attention sequence differ.

To reproduce a matrix substitution from an existing compiled stage:

```sh
python -m tools.build_nicolas_matrix_substitution \
  --stage "$COMPILED_STAGE" --out "$NEW_OUTPUT" \
  --npu-root "$NPU_ROOT" --rtl-root "$RTL_ROOT" \
  --riscv-root "$RISCV_ROOT"
```

Link the replacement catalog object with the same host stage and run with
the same Spike build and timing settings.
