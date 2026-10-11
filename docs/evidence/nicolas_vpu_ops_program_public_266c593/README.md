# Nicolas's complete VPU operation program through one public object

The [index](index.json) binds Nicolas's pinned `vpu_ops.c` and `vpu_ref.h` to
compiler revision `38ba247`, RTL revision `266c593`, and two MX+VPU Rocket
profiles. `tools.qualify_nicolas_vpu_ops_program` translates the source's 30
ordered VPU commands into one flat typed MLIR function. Its version 2 buffer
map requests a host snapshot after each source check and reloads A2 over A
after the write-after-read case. The object compiler emits all flush,
configuration, transfer, VPU, readout, and fence instructions; the driver has
no handwritten accelerator commands.

On Nicolas's pinned Spike, the generated program matches **13,056 BF16
values** in 30 snapshots against the source's bit-exact reference functions.
The 30 snapshots cover all **29 named source checks**: the source's combined
`dual X,Y` check appears as two snapshots. The two selected profiles each
pass. Their typed MLIR differs by profile hash, while the object, ELF, and
Spike log hashes agree. A second clean build under the E4M3-only profile
produced an identical receipt. The archive includes the MLIR, buffer map,
generated issuer and object, physical command stream, driver, ELF, logs,
and manifests for both profiles.

From a fresh compiler checkout with the pinned RTL, RISC-V tools, model2MLIR
Python environment, and native `mx-gemmini-opt`:

```sh
python -m tools.qualify_nicolas_vpu_ops_program \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt "$MX_OPT" --out-dir "$OUT_DIR"
```

For the second run, add
`--profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json`.
The command refuses an existing output directory and checks the source,
oracle, profile, compiler object, and complete Spike output. This is
functional Spike evidence; RTL queue timing and FPGA behavior remain open.
