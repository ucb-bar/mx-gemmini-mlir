# Nicolas DRAM-mvout accumulator command objects

This archive, generated from compiler commit `ef679cf`, contains data-free
RV64 Rocket/RoCC issuer objects for the three
`*DRAMMvout.c` source geometries at Nicolas RTL revision
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`. Each typed BF16 readout
selects `source_memory = "accumulator"`; the source-bound lowering routes the
last compute result to address `0x80000000` with flag `0xb8`, then emits
accumulator MVOUTs with source-shaped addresses and output offsets. The
generated object contains no operand or golden data.

| Pinned source | Compute K waves | Accumulator MVOUTs | Object SHA-256 |
|---|---:|---:|---|
| FP8 64×64×64 | 1 | 4 | `d179e5bd407b1491df0197d013b7935a4a8562da3138eb7c63f9a305a18d2f7f` |
| FP4 64×64×64 | 1 | 4 | `55b05ae01a3eb4f545bec6e6e986b231aead311714077ec43407a0b21ac11a43` |
| FP8 128×128×256 | 2 | 16 | `6b89572a30debf5f1b88e17f627acee03a4e1e38622d2f97b7c16b3d2cffe80b` |

The `accumulator_command_audit.json` in each case pins the C source hash,
checks the compute destination and flags, and compares the MVOUT address and
output pointer sequence with the source geometry. Independent second builds
produced identical bound MLIR, physical program, issuer C, and object hashes
for all three cases.

**Qualification limit:** These are command-generation results, not numerical
hardware results. Nicolas's non-`SPIKE_SIM` source writes RoCC commands through
an MMIO gateway and fills the scale SRAM with constant E8M0 byte `0x7f`;
this Rocket/RoCC compiler path uploads the nonconstant scales from Nicolas's
source header. The 128×128×256 compiler plan also uses two K waves, where the
source issues one loop command. The pinned Spike extension cannot execute an
accumulator output address. The separate [Spike fallback
archive](../nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593/README.md)
checks the complete BF16 outputs for the scratchpad route only. No FPGA
numerical parity, cycle overlap, or exact source-hardware command equivalence
is claimed here.
The [controlled constant-scale candidate replay](../nicolas_accumulator_constant_scale_divergence_266c593/README.md)
quantifies the scale-value gap: 24,516 of 24,576 BF16 values differ from the
source-header goldens when the generated objects receive the hardware branch's
constant `0x7f` scale bytes under the isolated Spike model.

Reproduce each object with the pinned RTL, RISC-V toolchain, and built native
`mx-gemmini-opt`:

```bash
for case in fp8_64x64x64_dram_mvout_spike \
            fp4_64x64x64_dram_mvout_spike \
            fp8_128x128x256_dram_mvout_spike; do
  python -m tools.compile_nicolas_accumulator_readout \
    --case "$case" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
    --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR/$case"
done
```

The command refuses an existing output directory and checks the pinned source
hash before compilation. It reuses the archived pinned model2MLIR handoff and
source payload, so rerunning it does not recapture PyTorch.
