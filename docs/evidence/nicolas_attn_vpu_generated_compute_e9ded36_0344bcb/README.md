<a id="generated-compute-stages-for-nicolass-dense-attention"></a>

# Generated compute stages for the reference implementation’s dense attention

This replay uses the reference implementation’s `attn_vpu.c` at `npu-exploration` `0344bcb`, its
default 32×64×64 data, the current model2MLIR `e9ded36` capture, and the
`MxE4M3Fp4VpuGemminiRocketConfig` profile from RTL `266c593`.

The qualifier rebuilds the typed [VPU/requant middle](../nicolas_attn_vpu_generated_middle_e9ded36_0344bcb/README.md)
through the public object compiler. It also encodes the source's QK and PV
matrix packets as two data-free RV64 objects. The transformed source checker
calls all three generated issuers. It contains no handwritten `mm` helper,
VPU call, or `spad_requant` call. The archived ELF links those three objects.

Two runs on the reference implementation’s pinned Spike extension produced identical logs. The
score matches all 512 four-element words of the source mesh golden; the score,
BF16 probability, requantized probability, and PV output hashes match the
unmodified [source baseline](../nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/README.md).
The source checker prints `attn_vpu PASSED`.

The source C still loads Q, Kᵀ, V, and E8M0 scales into the accelerator,
and reads scratchpad results back. QK/PV packets are source-specific physical
lowering, while the vector/requant middle comes from typed MLIR. A complete
compiler-generated attention object still needs typed QK/PV matrix lowering,
transfer/readout lowering, and a single linked runtime ABI.

From the repo root, set `NPU_ROOT` to the pinned `npu-exploration` checkout,
`RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to the reference implementation’s MX extension, and `OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_attn_compute \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --out-dir "$OUT_DIR"
```

The [receipt](receipt.json) records source, compiler, RTL, tool, object, ELF,
and Spike hashes. Use `--mx-opt /path/to/mx-gemmini-opt` to run the native
dialect verifier during the object build.
