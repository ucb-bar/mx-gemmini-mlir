# Experimental Nicolas MX accumulator Spike replay

The three compiler-generated, data-free accumulator issuer objects in
[`nicolas_accumulator_readout_objects_266c593`](../nicolas_accumulator_readout_objects_266c593/README.md)
were linked with drivers containing Nicolas's pinned source data and run on a
**task-local patched Spike extension**. All 24,576 BF16 values matched the
source goldens: 4,096 for FP8 64³, 4,096 for FP4 64³, and 16,384 for FP8
128×128×256. The [index](index.json) pins source, object, extension, driver,
ELF, Spike, and log hashes; each case's `spike.log` records its full-output
check. Two fresh builds in distinct output directories produced identical
extension, driver, ELF, and Spike log hashes.

The [candidate patch](../../../tools/patches/nicolas_spike_mx_accumulator_candidate_266c593.patch)
changes only a copied `software/libgemmini` extension from Nicolas's pinned
`266c593f2cb51d7e3fe83fc0317072b585ac3c52` RTL checkout. It routes
accumulator-tagged BF16 readouts from Spike's existing MX shadow result to
DRAM using the source's FP8/FP4 output geometry and rejects unsupported
accumulator requests. The compiler object still issues the accelerator
commands; the driver contains no handwritten replacement accelerator kernel.

**Qualification boundary:** This is a numerical check of compiler addresses
under a candidate Spike model. It does not model RTL accumulator packing,
queue timing, the MMIO gateway, or the source hardware path's constant `0x7f`
scale initialization. Stock Spike cannot execute this accumulator path. The
compiler object uses RoCC scale uploads and, for 128×128×256, a two-wave K
schedule. No RTL, FPGA, exact source-hardware command, or timing parity is
claimed.

The [constant-scale replay](../nicolas_accumulator_constant_scale_divergence_266c593/README.md)
uses the same generated objects and candidate extension but changes their
runtime A/B scales to the `0x7f` values written by Nicolas's hardware branch.
It diverges from the source-header goldens in 24,516 of 24,576 BF16 values.

To replay from a fresh checkout with the pinned RTL and RISC-V tools:

```bash
python -m tools.qualify_nicolas_accumulator_candidate \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --out-dir "$OUT_DIR/nicolas-accumulator-candidate"
```

The tool refuses to overwrite an output directory, checks the pinned RTL
source hashes and archived object manifests, applies the patch to a copy,
builds the Spike extension, links all three generated objects, and checks
every BF16 output. It leaves the Nicolas RTL checkout untouched.
