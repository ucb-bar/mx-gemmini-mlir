# Nicolas MX scale-memory selector RTL probe

The [qualifier](../../../tools/qualify_nicolas_scale_mem_rtl.py) copies
`ScaleFactorMem.scala` byte-for-byte from Nicolas's `gemmini-mx-cleanup`
revision `266c593` into an isolated Chisel 6.6 test project. It pins the
original module, its interface source, the MxGen Mill launcher, and the two
test inputs by SHA-256. Minimal bundle declarations replace Chipyard-only
interface dependencies; the scale-memory module itself is unmodified.

Two tests load distinct E8M0 byte values into scale-memory halves and check
the actual combined-scale output for independent activation and weight
selectors. The DIM16 test checks all four selector combinations, then switches
back to half zero, resetting the module between cases. The four-lane test
fills both banks per half and advances through complete scale-row cycles
without reset between selector changes. Both use a depth-16, 128-bit SRAM
configuration. The expected sums are 11, 14, 41, 44, then 11.

This is RTL module evidence for bank routing and E8M0 scale addition. It does
not prove that `ExecuteController` latches command bits 60 and 61 correctly
under traffic, that DMA lands the next half in time, or that the full FP6
contraction runs on RTL or FPGA. The compiler's archived 16-wave FP6 physical
program alternates selector bits `00` and `11`; the regression test links
that command encoding to these module checks without claiming full-system
qualification.

Reproduce with:

```sh
python -m tools.qualify_nicolas_scale_mem_rtl \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --out-dir /new/scale-half-rtl-probe
```
