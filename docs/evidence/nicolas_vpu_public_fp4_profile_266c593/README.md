# Nicolas VPU public objects on the FP4-capable profile

The fourteen model2MLIR VPU captures from the
[E4M3-only public-object replay](../nicolas_vpu_public_objects_266c593/README.md)
were rebound to Nicolas's `MxE4M3Fp4VpuGemminiRocketConfig` profile. The
binding functions reproduced each original bound MLIR byte for byte when
given the original profile, then emitted profile-specific MLIR for the
FP4-capable build. Native `mx-gemmini-opt` and the public `tools.compile_object`
accepted all fourteen rebound modules.

Every generated object ran on pinned Spike and matched Nicolas's `vpu_ref.h`:
**6,016 primary BF16 outputs and 128 EXPSUM outputs**, with no mismatches.
Two independent builds produced identical report contents. Each case's
issuer object, linked ELF, and Spike log is byte-identical to the E4M3-only
profile result, while all fourteen bound MLIR digests and their profile
digests differ. The [index](index.json) records both sets of hashes; the case
directories contain the rebound MLIR, binding manifest, physical program,
object manifests, and Spike log. The identical object and ELF files remain in
the [E4M3-only archive](../nicolas_vpu_public_objects_266c593/).

Reproduce with the pinned RTL and RISC-V tools:

```bash
python -m tools.qualify_nicolas_vpu_public_objects \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt "$MX_OPT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out-dir "$OUT_DIR/nicolas-vpu-fp4-capable"
```

This qualifies the selected **BF16 VPU operations** on a profile that also
has FP4-capable MX hardware. It does not establish FP4 VPU operand semantics,
RTL queue timing, or execution of the complete `vpu_ops.c` harness.
