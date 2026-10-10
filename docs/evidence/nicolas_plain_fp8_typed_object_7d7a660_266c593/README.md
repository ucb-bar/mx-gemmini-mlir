# Nicolas FP8 128³ from typed MLIR to a public MX object

Commit `7d7a660b8bfe6911904ae9585f47974bc83d814f` captured a
`torch.matmul` at shape 128×128×128 with model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`. It bound the packed FP8
operands, E8M0 scales, and BF16 reference from Nicolas's checked-in
`matmul_tiled_fp8_128x128.c` and `matmul_fp8_128x128.h`, then used
`tools.compile_object` to emit a linkable RV64 RoCC object. The data-free object
was linked with a driver that only passes the source arrays, fences, and
compares all 16,384 BF16 outputs against the source header. Pinned Spike
reported **zero mismatches**.

This demonstrates numerical regeneration of this one source test's matrix
operation. It does not compile the C program's performance instrumentation,
warm-cache experiment, or all 171 Nicolas MX programs. The artifact uses the
MX-only `MxGemminiRocketConfig`; no VPU is involved. The pinned source revision
is `266c593f2cb51d7e3fe83fc0317072b585ac3c52` with software gitlink
`350547f9843f46f485d4d1dd4c20b2f52f4844bf`.

The archived `receipt.json` binds source, frontend, profile, object, ELF, and
Spike hashes. `bundle/` contains the exact source operands and golden,
`payload_bound.mlir` is the typed executable input, and `object/` includes the
physical program and generated issuer. `run/mx_driver.c` contains no MX command
schedule. A second fresh run produced identical object, physical-program, ELF,
and Spike-log SHA-256 hashes.

Reproduce from this compiler commit, the above pinned RTL with its submodules,
model2MLIR and MXQuant revisions in the receipt, a native `mx-gemmini-opt`,
and RISC-V GCC/Spike:

```sh
"$PYTHON" -m tools.qualify_nicolas_plain_fp8_object \
  --model2mlir-root "$MODEL2MLIR_ROOT" --mxq-root "$MXQUANT_ROOT" \
  --rtl-root "$MX_RTL_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
  --out-dir "$OUT_DIR"
```

`PYTHON` must have PyTorch, model2MLIR's dependencies, and the MX compiler's
Python dependencies installed. The command refuses an existing output directory.
