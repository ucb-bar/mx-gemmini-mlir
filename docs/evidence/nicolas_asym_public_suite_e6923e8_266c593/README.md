# All pinned Nicolas asymmetric sources through public MX objects

Compiler commit `e6923e8a774748f55ea3d065f6d3832d6618022d` replayed every
`matmul_tiled_asym_*` program in Nicolas's pinned MX Makefile roster:
**74 of 74 named C programs** and **360,960 BF16 output comparisons** passed
on pinned Spike. The six groups cover DIM8, DIM16, and DIM32 64×64 sources,
the 16×32 single-tile source, and five larger 128×128 sources. The
`index.json` checks exact source hashes against `source_inventory_baseline.json`.

Each case captures a PyTorch matmul with pinned model2MLIR, binds the packed
operand, scale, and LUT arrays from Nicolas's source header, lowers typed MLIR
to physical MX commands, emits a **data-free RV64 RoCC object** through
`tools.compile_object`, links it into a generated checker, and compares every
BF16 output with the checked-in source golden. The public issuer C and physical
program must match the preexisting asymmetric lowerer byte for byte before the
object is linked. This reuses that lowerer; the public path adds object
packaging and an explicit pointer ABI.

This archive contains the receipt, object, physical program, ELF, and Spike
log for every program. One case from each group also retains frontend and
bound MLIR, the recipe, input and golden bytes, generated driver, and issuer
C. The object manifests show zero embedded operand or golden bytes and zero
allocated data-section bytes.

Reproduce from the compiler commit above with Gemmini RTL at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR at
`e9ded36eb85abf2d9097ac4dc11457c825853388`, MXQuant at
`b4af5430bac147f4a16126931cc0177367cc3982`, native
`mx-gemmini-opt`, and RISC-V GCC/Spike:

```bash
"$PYTHON" -m tools.qualify_nicolas_asym_public_suite \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --jobs 4 --out-dir "$OUT_DIR"
```

The suite covers the numerical results of these 74 named source programs.
It does not compile their C performance instrumentation, prove FPGA timing,
or qualify the MX+VPU Radiance configuration for every asymmetric pair.
Generated legal mode headers outside the Makefile roster are tracked by the
separate mode qualification evidence.
