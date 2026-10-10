# Nicolas dual-layout FP4 scratchpad requant through the public object compiler

This replay uses Nicolas's pinned `spad_requant_fp4.c` input generator and
per-element reference checks. The typed MLIR represents one 64×128 BF16 input
and two `mx_gemmini.spad_requant` operations: flat and operand-A-tiled FP4
output. `tools.compile_object` selects the physical lowerer and emits a
data-free, five-buffer RV64 RoCC object. The compiler driver keeps the source
input and reference code but calls that object for both accelerator operations.

The source and compiler-linked executables both passed on the pinned Spike:
**16,384 FP4 codes and 512 E8M0 scales** were compared with zero mismatches.
The object has 101 commands, one fence, and no embedded operand or golden data.
Its SHA-256, `ab3975ad7ad33a913ceadc7858d3ab934ab1b41a5aa16f54bb9be8b733e7b067`,
matches the earlier diagnostic object's bytes. The two cycle counts in the logs
come from different issue sequences and are not a performance-parity claim.

The graph is rendered from checked source geometry and bindings by
`render_fp4_dual_requant`. It is **not** a model2MLIR capture of the C program.
The driver patch and both original/compiled Spike logs are archived here so
that distinction is reviewable.

From the repository root, with the pinned RTL checkout, RISC-V toolchain, and
built native verifier:

```bash
python -m tools.qualify_nicolas_fp4_dual_public_object \
  --rtl-root "$MX_GEMMINI_RTL_ROOT" \
  --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /tmp/nicolas-fp4-dual-replay \
  --baseline-receipt \
    docs/evidence/nicolas_fp4_dual_public_object_d4ed0d8_266c593/receipt.json
```

The replay refuses changed source/header hashes, the wrong RTL revision,
unsupported profiles or altered graph bindings. The baseline comparison checks
every recorded digest, including both ELF and Spike-log hashes. The compiler
revision used to create this archive is
`d4ed0d8e0fea4acb089a198f7fec11dbd03f67f1`.
