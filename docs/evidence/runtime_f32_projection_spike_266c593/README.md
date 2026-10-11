# Live f32 MX projection qualification

Two small-LLaMA FP4 MLP sites captured through model2MLIR were replayed on
Nicolas's `gemmini-mx-cleanup` Spike extension at RTL revision `266c593`.
The RV64 image reads the archived **f32 source operands**, pads them to legal
MX geometry, quantizes them at runtime with `runtime_quant.c`, calls the
data-free compiler-generated MX issuer, and widens the BF16 readout to f32.

| Site | Logical M×N×K | Padded M×N×K | BF16 words checked | Result |
| --- | --- | --- | ---: | --- |
| Gate (`g`) | 8×344×128 | 32×352×128 | 11,264 | Exact match |
| Down (`dn`) | 8×128×344 | 32×128×352 | 4,096 | Exact match |

Each run compared all runtime-packed code and E8M0 scale bytes with the
independent archived MXQuant payload, then compared every BF16 output with
the archived arithmetic reference. The widened f32 words were also checked
against the BF16 bit patterns. Both Spike processes exited zero. See
[`g.json`](g.json) and [`dn.json`](dn.json) for source, profile, generated
issuer, ELF, and simulator-log hashes.

This qualifies two separately invoked projections. It does not cover Merlin's
dense catalog ABI, a connected MLP graph, checkpoint inference, or model state.

Reproduce each site from the repository root with the pinned RTL and RV64
toolchain selected explicitly:

```sh
python -m tools.qualify_runtime_f32_projection_spike \
  --site g --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools --out-dir /new/g-output
python -m tools.qualify_runtime_f32_projection_spike \
  --site dn --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/riscv-tools --out-dir /new/dn-output
```
