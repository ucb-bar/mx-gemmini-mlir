# One RV64 object for Nicolas's dense MX attention commands

The pinned source is `npu-exploration` `0344bcb`'s 32×64×64 `attn_vpu.c`.
The vector/requant stage is bound to the current model2MLIR `e9ded36` graph;
the surrounding source-specific QK/PV and I/O packets are assembled into one
127-item physical program. The compiler emits one data-free RV64 object with
12 explicit runtime pointer arguments. The archived ELF links this one
object with the source's correctness checker and deterministic generated data.

Nicolas's pinned Spike extension ran the ELF twice with identical logs.
The score matches all 512 four-element words of the mesh golden, and score,
probability, requantized probability, and PV output hashes match the
[unmodified source baseline](../nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/README.md).
The checker prints `attn_vpu PASSED`. The physical program, issuer C/object,
driver, ELF, log, and digest receipt are archived here.

The host checker still issues `gemmini_flush` and `gemmini_fence` for setup and
validation. The QK/PV and I/O packet choices are pinned to this source; a
single typed MLIR graph for all stages is the next compiler gate. The replay
does not qualify other attention shapes or models.

From the repo root, set `NPU_ROOT` to the pinned `npu-exploration` checkout,
`RTL_ROOT` to `gemmini-mx-cleanup` `266c593`, `RISCV_ROOT` to the RV64 toolchain,
`SPIKE_EXTENSION` to Nicolas's MX extension, and `OUT_DIR` to a fresh directory:

```sh
python -m tools.qualify_nicolas_attn_compute \
  --npu-root "$NPU_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --spike-extension "$SPIKE_EXTENSION" --single-object --out-dir "$OUT_DIR"
```

The [receipt](receipt.json) records source, frontend, compiler, RTL, tool,
object, ELF, and Spike hashes. `--mx-opt` verifies the typed vector stage with
the native dialect verifier during compilation.
