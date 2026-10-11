# Nicolas dependent VPU chain through the public object compiler

Nicolas's pinned bareMetalC/vpu_ops.c contains an unfenced
ADD → MULS(0.5) → RMAX scratchpad chain. The source run checks its final
16 rows against vpu_ref.h as one of 29 VPU checks. This archive captures
the corresponding PyTorch graph with upstream model2MLIR e9ded36, binds
the three operations to typed MX MLIR, and compiles them through
tools.compile_object to one data-free RV64 RoCC object.

On Nicolas's pinned Spike, the generated object matches all **512 BF16**
values at the intermediate MULS result and all **128 BF16** values at the
final RMAX result. Both MxE4M3VpuGemminiRocketConfig and
MxE4M3Fp4VpuGemminiRocketConfig pass. Their profile-bound MLIR differs,
while issuer object, ELF, and Spike output hashes agree. A second
independent build of the first profile produced an identical receipt. The
[index](index.json) records source, frontend, profile, tool, object, ELF,
and log hashes; the object, issuer, physical stream, ELF, and logs are
archived for both profiles.

Use a clean source snapshot of upstream model2MLIR e9ded36. The qualifier
checks the full m2m Python source closure, so a dirty checkout cannot
silently change the capture:

~~~sh
python -m tools.qualify_nicolas_vpu_chain_public \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$NEW_OUTPUT_DIR"
~~~

For the FP4-capable VPU config, add
--profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json.
The replay produces frontend.mlir, bound.mlir, abi.json,
object/mx_issue.o, vpu_chain.elf, spike.log, and receipt.json in a new
directory.

This result qualifies the **three-op chain**, including its scratchpad
dependency and runtime-buffer ABI. It does not compile all 29 checks or
the host control flow of vpu_ops.c as one MLIR program. RTL cycle timing
and FP4 VPU operand semantics remain separate.
