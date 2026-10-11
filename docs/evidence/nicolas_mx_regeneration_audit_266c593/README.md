# Nicolas MX program-level regeneration audit

The [index](index.json) classifies every program in Nicolas's pinned
171-entry MX Makefile roster at RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. The report validates the
archived source binding, generated object or executable evidence, Spike outcome,
and recorded output-comparison count for **156 matrix/LUT programs** and
**six connected FP4/FP6/FP8 matrix chains** with selected-path full-result
receipts. For the six chains it validates both C1 and C2 code and scale
comparisons and the saved ELF and Spike log. FP8 receipts also pin the
archived reference arrays. Each row names its exact receipt and available
frontend, compiler, and target profile revisions.

The other **nine entries** are explicitly listed rather than counted as failed
compilations. Seven are VPU, scratchpad requantization, connected-chain, or
memory programs with separate evidence and different qualification scopes.
The two remaining debug programs, `matmul_ws_mx_generic` and
`matmul_single_tile_test`, have no direct source-hash receipt. In
the latter source, the golden-comparison loop
is commented out, so its printed PASS line alone is not a numerical oracle.

This audit does **not** claim that the compiler regenerated every instruction
or the complete C control flow of the 162 selected paths. The archived
FP8 128×128 chain contains a generated ELF and pinned object hashes, while
the individual object files are not archived in that original bundle. It does
not upgrade patched Spike, source-only runs, RTL timing, or FPGA behavior into a
compiler qualification. The separate evidence for the other seven entries must
be reviewed at its stated scope before making a program-level claim.

Rebuild and check the inventory and audit against the pinned RTL checkout:

```bash
python -m tools.audit_nicolas_mx_roster \
  --rtl-root "$MX_RTL_ROOT" \
  --out docs/evidence/nicolas_mx_source_inventory_266c593/index.json --check
python -m tools.audit_nicolas_regeneration --check
```
