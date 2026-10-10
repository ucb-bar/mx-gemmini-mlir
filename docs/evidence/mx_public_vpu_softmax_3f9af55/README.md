# Public MX object compiler: BF16 VPU softmax

The `3f9af55` compiler revision captured a 16×32 BF16 softmax from model2MLIR,
bound its six VPU operations to both Nicolas DIM16 VPU Rocket profiles, and
compiled the typed module through `python -m tools.compile_object`. The output
is a linkable RV64 RoCC `mx_issue.o` with two pointer arguments and no embedded
input or golden data. The qualifier linked that object with Nicolas's original
input generator and bit-exact checker. Pinned Spike compared all 512 BF16
outputs: zero mismatches for both profiles.

The `e4m3_fp4_vpu/` and `e4m3_vpu/` directories contain the frontend and bound
MLIR, ABI, physical command stream, generated issuer and object, manifests, and
Spike output. Each `artifact_manifest.json` records tool and source revisions,
hashes, and the ELF hash. The selected RTL revision is
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`; model2MLIR is
`e9ded36eb85abf2d9097ac4dc11457c825853388`. This is Spike evidence for
the source softmax shape and the two named profiles; it is not FPGA evidence or
general VPU graph coverage.

From this compiler revision, with those checkouts and the pinned RISC-V tools:

```sh
PYTHONPATH="$MODEL2MLIR_ROOT:$MXQUANT_ROOT:." "$PYTHON" \
  -m tools.qualify_nicolas_vpu_softmax \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "$OUT_DIR"
```

Use `MxE4M3VpuGemminiRocketConfig.json` for the second profile. The command
requires the model2MLIR Python environment, the RTL submodules, GCC, Spike,
and a native `mx-gemmini-opt` build. It refuses an existing output directory.
