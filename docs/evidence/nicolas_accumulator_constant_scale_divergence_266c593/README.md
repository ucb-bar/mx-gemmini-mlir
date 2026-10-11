# Nicolas accumulator scale-value divergence

Nicolas's three `*DRAMMvout.c` hardware branches write constant E8M0 byte
`0x7f` into the MX scale SRAM. Their source headers contain nonconstant A/B
scales, and the source BF16 goldens correspond to those header values. The
compiler-generated accumulator objects accept scale buffers at runtime. This
experiment passed constant `0x7f` buffers to the **same objects** and replayed
them on the [isolated candidate Spike extension](../nicolas_accumulator_candidate_spike_266c593/README.md).

| Source geometry | BF16 mismatches against header golden | Values checked |
|---|---:|---:|
| FP8 64×64×64 | 4,096 | 4,096 |
| FP4 64×64×64 | 4,036 | 4,096 |
| FP8 128×128×256 | 16,384 | 16,384 |

The [index](index.json) pins the unchanged compiler object and Spike extension
hashes, constant-scale driver and ELF hashes, and the three logs. Two builds in
different output directories produced identical receipts. Rerunning the
original header-scale mode after this change reproduced its earlier index
exactly: **zero mismatches** across the same 24,576 outputs.

This establishes a numerical divergence **under the candidate Spike model**
when only the scale values change. It does not show actual RTL or FPGA output.
The constant values match the used scale bytes in Nicolas's hardware source;
the compiler still uploads them through RoCC rather than the source's MMIO
gateway. The source's hardware path also has different scheduling. A source
golden based on header scales therefore cannot be used as proof that the
constant-scale hardware branch computes the same values.

Reproduce with the pinned RTL and RISC-V tools:

```bash
python -m tools.qualify_nicolas_accumulator_candidate \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --scale-mode hardware_constant_0x7f \
  --out-dir "$OUT_DIR/nicolas-accumulator-constant-scales"
```
