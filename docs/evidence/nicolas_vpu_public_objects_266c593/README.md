# Nicolas VPU operations through the public object compiler

Fourteen typed MX VPU operations now compile through `tools.compile_object`
into **data-free RV64 Rocket/RoCC objects**. The twelve base operations use
the model2MLIR captures and bound MLIR in
[`nicolas_vpu_elementwise_compiled_266c593`](../nicolas_vpu_elementwise_compiled_266c593/),
and EXPSUB/EXPSUM use the corresponding
[`nicolas_vpu_fused_compiled_266c593`](../nicolas_vpu_fused_compiled_266c593/)
captures. Every bound module passed the newly built native `mx-gemmini-opt`
verifier. The public compiler selected the `vpu_elementwise` family, generated
the transfers, VPU command, and readout, then emitted a linkable object.

All fourteen objects were linked to host drivers that generate Nicolas's
pinned inputs and check against `vpu_ref.h`. Pinned Spike reports **zero
mismatches over 6,016 primary BF16 outputs and 128 EXPSUM outputs**. The
twelve base issuers match the older specialized compiler issuers byte for
byte. For EXPSUB/EXPSUM, the public lowerer loads only the one broadcast B
group required by the typed operation; the older specialized issuer loaded
all four groups. Both public fused objects pass their complete source
reference checks. The EXPSUB host driver drops an unused fourth pointer to
match its three-slot public ABI; its numerical oracle is unchanged.

The [index](index.json) pins the source, profile, compiler, toolchain, Spike,
object, ELF, and log hashes. Each case directory includes its generated C
issuer, object, physical program, object manifests, ELF, and Spike log.
Two independent builds produced identical complete index contents, including
every object, ELF, and log hash. The replay command is:

```bash
python -m tools.qualify_nicolas_vpu_public_objects \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt "$MX_OPT" --out-dir "$OUT_DIR/nicolas-vpu-public"
```

This qualifies the 14 VPU operation classes as separate generated programs
on the selected DIM16 `MxE4M3VpuGemminiRocketConfig` Spike model. It does not
compile the entire `vpu_ops.c` test harness as one MLIR graph, establish RTL
queue timing, or infer support in profiles without the relevant VPU units.
