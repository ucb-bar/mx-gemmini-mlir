# Connected VPU dispatch optimization on Spike

A PyTorch BF16 module computes `(x * 1.1015625) + 0.333984375`. The latest
model2MLIR main checked for this run, `905d2c9`, captures the scalar operations
and the intermediate BF16 rounding. Merlin outlines each complete source
operation, including its casts. The MX adapter reads the typed function bodies
to select `muls` and `adds` and encode their scalar immediates. No node name or
position chooses the arithmetic.

Merlin's generic connected-dispatch pass selects the private intermediate.
The MX backend checks matching physical rows, dtype, shape, and other loads,
then removes that intermediate's DMA store and reload. The two arithmetic
commands and synchronization are unchanged. RV64 GCC builds the command issuer
and the identical input/checker/timer driver for each variant. Both generated
MX modules also pass the native `mx-gemmini-opt` verifier.

| BF16 elements | Baseline warm modeled cycles | Connected warm modeled cycles | Reduction | Removed intermediate traffic per call |
| ---: | ---: | ---: | ---: | ---: |
| 136, including a partial DMA block | 291 / 291 | 192 / 192 | 34.0% | 544 bytes |
| 512 | 623 / 623 | 402 / 402 | 35.5% | 2,048 bytes |

Each program runs once cold and twice warm. Every output bit matches the
captured PyTorch golden and an independent scalar BF16 oracle in all three
runs. The fixture includes values whose results would change if the
intermediate rounding were removed. The measured extension load bytes fall
by half; static command inspection confirms that only the intermediate DMA
commands disappear. Production selection, using the measured positive
savings, regenerates the same qualified command stream.

These are small connected VPU regions on Nicolas's `e4m3_vpu` Spike timing
model, with `GEMMINI_MODE=both` and `mem.host_tracking=0`. They establish an
executable first integration of the shared pass. They do not measure a full
model, matrix-to-VPU fusion, regular Gemmini, or FPGA latency. Current full
model f32 boundaries and matrix/requantization chains need further integration.

Run the portable archive verifier without a checkpoint or simulator:

```sh
python -B docs/evidence/connected_dispatch_20261016/verify.py
```

To regenerate a case, use the recorded profile, source-bound RTL checkout,
RV64 toolchain, Nicolas extension, and a Python environment with PyTorch,
model2MLIR, Merlin, and xDSL 0.65. `STARTUP_DIR` is the matching baremetal
`riscv-tests/benchmarks/common` directory with its environment headers.

```sh
PYTHONPATH="$MERLIN_ROOT/src:$MX_ROOT" python -m tools.qualify_connected_dispatch_spike \
  --model2mlir-root "$MODEL2MLIR_ROOT" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --startup-dir "$STARTUP_DIR" \
  --riscv-root "$RISCV_ROOT" --extlib "$EXTENSION" \
  --rows 64 --out-dir "$OUT_DIR"
```

The archive keeps source and outlined MLIR, the dispatch graph, both plans,
actual input and golden bytes, native verification records, generated C,
physical commands, Spike logs, counters, and hashes. Objects and executables
remain build artifacts; the receipts record their hashes.
