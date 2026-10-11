# Nicolas DRAM-mvout source programs: Spike fallback

The three pinned `*DRAMMvout.c` programs have two conditional execution
paths. Under `SPIKE_SIM`, each computes BF16 C into scratchpad row zero and
reads it through `gemmini_mx_read_smem`. Under the hardware path, each selects
accumulator address `0x80000000`, skips the scratchpad store, and issues
accumulator-to-DRAM `gemmini_mvout` commands. Nicolas's pinned Spike extension
does not model the latter path.

The source-bound compiler now emits a data-free RV64 RoCC object for the
**Spike fallback** of each program. A fresh PyTorch → model2MLIR capture
provides the contraction; the checked source headers provide packed operands,
scales, and the BF16 golden. The compiler stores C at scratchpad row zero and
reads it through RoCC MVOUT. The original source uses a CPU scratchpad read,
so transport differs. Both paths independently match every golden BF16 value
on Nicolas's pinned Spike:

| Source | BF16 values | Compiled object | Original source |
|---|---:|---|---|
| `matmul_tiled_fp8_64x64_DRAMMvout.c` | 4,096 | 0 mismatches | Passed |
| `matmul_tiled_fp4_64x64_DRAMMvout.c` | 4,096 | 0 mismatches | Passed |
| `matmul_tiled_fp8_128x128x256_DRAMMvout.c` | 16,384 | 0 mismatches | Passed |

The two-wave 128×128×256 schedule retains the same zero C destination across
both K waves. The [per-case command audits](fp8_128x128x256_dram_mvout_spike/smem_readout_equivalence.json)
check that destination, all readout rows, and flat output offsets. The receipts
mark `hardware_accumulator_mvout_qualified=false`. These results establish
numerical parity for `SPIKE_SIM` only; they do not validate the hardware
accumulator-to-DRAM command path, timing, or FPGA behavior.

Reproduce from compiler commit `6ed3fcb`, RTL
`266c593f2cb51d7e3fe83fc0317072b585ac3c52`, model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388`, and MXQuant
`b4af5430bac147f4a16126931cc0177367cc3982`:

```bash
for case in fp8_64x64x64_dram_mvout_spike \
            fp4_64x64x64_dram_mvout_spike \
            fp8_128x128x256_dram_mvout_spike; do
  "$PYTHON" -m tools.qualify_nicolas_plain_matrix_object \
    --case "$case" --model2mlir-root "$MODEL2MLIR_ROOT" \
    --mxq-root "$MXQUANT_ROOT" --rtl-root "$MX_RTL_ROOT" \
    --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
    --riscv-root "$RISCV_ROOT" --mx-opt build/tools/mx-gemmini-opt \
    --out-dir "$OUT_DIR/$case"
done
```

Each run refuses to overwrite its output directory.
