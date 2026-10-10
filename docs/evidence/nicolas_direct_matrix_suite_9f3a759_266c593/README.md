# One-command replay of direct Nicolas matrix object cases

Commit `9f3a759` added `tools.qualify_nicolas_plain_matrix_suite`, which ran
all 11 registered direct Nicolas matrix cases from fresh PyTorch/model2MLIR
captures through source payload binding, the public `tools.compile_object`
path, RV64 linking, and pinned Spike. **All 11 passed**. The suite index
records **111,872 checked items**: BF16 values for plain readout cases, and
FP8 codes or packed FP4/FP6 bytes plus E8M0 scales for quantized cases. This
is a sum of comparison items with different storage units, not a count of
homogeneous tensor elements.

The suite checks each child's source hashes, selected profile and mesh,
expected comparison extent, compiler and tool revisions, object and ELF
hashes, and full-output Spike result before recording success. Its
`index.json` and 11 child receipts are archived here. The corresponding
source bundles, typed MLIR, objects, ELFs, and Spike logs remain in the
[per-case archives](../../nicolas_mx_kernel_coverage.md). The new suite's
object and Spike-log hashes match those archives for every case.

This is a reproducible gate for the named direct matrix cases. It does not
compile the remaining Nicolas programs, C performance instrumentation, or
arbitrary source control flow. The pinned source roster contains 171 MX test
programs, so suite success is deliberately narrower than full roster
coverage.

Reproduce from the compiler commit above with the pinned RTL/submodules,
model2MLIR/MXQuant revisions in `index.json`, native `mx-gemmini-opt`, and
RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_matrix_suite \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
