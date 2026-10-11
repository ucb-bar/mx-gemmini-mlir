# Nicolas MX program-level regeneration audit

The [index](index.json) classifies every program in Nicolas's pinned
171-entry MX Makefile roster at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. The report validates the
archived source binding, generated object or executable evidence, Spike outcome,
and recorded output-comparison count for **156 matrix/LUT programs**,
**six connected FP4/FP6/FP8 matrix chains**, and **eight specialized programs**
with selected-path full-result receipts. The specialized programs are
`vpu_softmax`, `spad_requant`, `spad_requant_fp4`, `chain_pipelined`,
`chain_vpu_spad_requant`, `mx_mem_bw`, `vpu_ops`, and `matmul_ws_mx_generic`;
the audit checks their archived
generated object, Spike log, source binding, and complete output count.
The memory benchmark's full
16 KiB readout qualifies bytes, not timing or performance counters.
For the six chains it validates both C1 and C2 code and scale
comparisons and the saved ELF and Spike log. FP8 receipts also pin the
archived reference arrays. Each row names its exact receipt and available
frontend, compiler, and target profile revisions.

The generic FP6 debug source uses a fixed MMIO gateway. Its compiler replay
qualifies the header's BF16, packed, and scale values through a Rocket object
captured with current model2MLIR; it does not qualify the original MMIO issue
sequence. The **one remaining entry**, `matmul_single_tile_test`, has no direct
source-hash receipt. Its golden-comparison loop is commented out, so its
printed PASS line alone is not a numerical oracle.

This audit does **not** claim that the compiler regenerated every instruction
or the complete C control flow of the 170 selected paths. The `vpu_ops`
qualification compiles all 30 ordered source VPU commands into one executable
and compares 13,056 BF16 values across 29 named source checks under both VPU
profiles; its CPU input and reference logic remains in a thin driver. The archived
FP8 128×128 chain contains a generated ELF and pinned object hashes, while
the individual object files are not archived in that original bundle. It does
not upgrade patched Spike, source-only runs, RTL timing, or FPGA behavior into a
compiler qualification.

Rebuild and check the inventory and audit against the pinned RTL checkout:

```bash
python -m tools.audit_nicolas_mx_roster \
  --rtl-root "$MX_RTL_ROOT" \
  --out docs/evidence/nicolas_mx_source_inventory_266c593/index.json --check
python -m tools.audit_nicolas_regeneration --check
```
