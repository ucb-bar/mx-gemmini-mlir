# Current model2MLIR connected MX chain qualification

The [suite receipt](index.json) covers Nicolas's FP8, FP4, and FP6 connected
matrix programs at both 64³ and 128³. A fresh PyTorch two-matmul graph was
captured with the clean committed model2MLIR source closure at `a042643`,
bound to each pinned C driver and header, lowered through the public
`tools.compile_object` command, linked as a standalone RV64 Rocket/RoCC
program, and run on Nicolas's pinned Spike. All six generated programs
matched every source-header C1 and C2 code and E8M0 scale: **122,880 codes
and 3,840 scales** across the suite. The generated objects contain no operand
or golden data.

The FP8 64³ object is byte-for-byte equal to the previously qualified public
object. The FP4 and FP6 physical command lists at both sizes equal the earlier
source-qualified streams. The FP8 128³ generated object also passes all four
full-output checks; this receipt does not assert command-byte identity with
the original C driver. A second fresh run produced the same suite receipt
and 51 selected artifacts byte-for-byte.

The [archive manifest](archive_manifest.json) pins input resources, frontend
captures, typed MX graphs, issuer sources and objects, linked ELFs, and Spike
logs. Check the archive with:

```sh
python -m tools.archive_nicolas_portable_chains \
  --out-dir docs/evidence/nicolas_portable_chains_a042643_266c593 --check
```

Recreate it from pinned tool and source checkouts with:

```sh
python -m tools.qualify_nicolas_portable_chains \
  --model2mlir-root /path/to/clean/model2MLIR-a042643 \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /new/mx-portable-chains
```

This qualifies the six named source shapes on Rocket Spike. Other connected
graph structures, mixed MX+VPU scheduling, RTL timing, and FPGA MMIO
transport remain separate gates.
