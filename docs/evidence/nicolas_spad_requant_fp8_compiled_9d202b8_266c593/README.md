# Nicolas flat and tiled FP8 scratchpad requantization

Compiler revision `9d202b80b4876af1d3e078c7a545f69e0e215277` lowers two
typed `mx_gemmini.spad_requant` operations over the same 32×64 BF16 input.
It emits a data-free RV64 RoCC object with input transfer, flat E4M3
requantization and readout, tiled E4M3 requantization and readout, and a
completion fence. The source-derived driver retains Nicolas's input generator
and bit-exact `mx_e4m3_ref.h` checker; its handwritten accelerator commands
are removed. The archived patch shows every replacement.

The original source ELF and generated compiler ELF each pass **4,096 E4M3
code comparisons** and **128 E8M0 scale comparisons** across the two layouts
on pinned Spike. Both Nicolas DIM16 VPU profiles pass independently:

| Profile | Archive |
|---|---|
| `MxE4M3VpuGemminiRocketConfig` | `e4m3_vpu/` |
| `MxE4M3Fp4VpuGemminiRocketConfig` | `e4m3_fp4_vpu/` |

The typed graph is bound to source geometry and source/header hashes. This
synthetic C test does not have a PyTorch/model2MLIR frontend capture. The
comparison qualifies numerical output on Spike, not the source's cycle count
or FPGA timing.

Reproduce from the compiler revision above and RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`:

```bash
"$PYTHON" -m tools.qualify_nicolas_spad_requant_fp8 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json \
  --out-dir "$OUT_DIR"
```

Select `MxE4M3Fp4VpuGemminiRocketConfig.json` for the second profile. The
command refuses to overwrite an existing output directory.
