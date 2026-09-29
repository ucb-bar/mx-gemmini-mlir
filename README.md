# MX Gemmini layout prototype

This out-of-tree compiler prototype is scoped to
`GemminiMxFPConfigs.standaloneMxFPConfig` at Gemmini
`f0167390b56fb315deea90ac1fc3983772e92d82` and MxGen
`a27ce3cd81513210c21f971ec3977defd13fa21e`. It carries no
Merlin provider registration or executable backend. Its layout rules are
source-derived and remain unreviewed until a matching simulator compares the
written bytes and command stream.

`mx_gemmini_support.layout` implements:

- `CONFIG_EX` activation/weight format codes;
- DIM16 logical scale-row ordering and format-specific physical bank address
  decoding;
- first-buffer E8M0 scale payloads and funct-27 `rs2`;
- capacity-bounded K-wave partitioning and zero-based scale payload slicing;
- `CONFIG_SCALE_MEM` `rs1` bitfield encoding;
- FP6 E3M2 codebook bit packing, 8-byte DMA padding, and funct-29 `rs2`.

Each planned wave fits the first 4 KiB active window for both operands and
the 9-bit `CONFIG_SCALE_MEM` K bound. This is a layout/capacity plan only:
the caller still must order scale uploads, configure the wave's loop bounds,
execute the corresponding operand tiles, and preserve the BF16 accumulator
across K waves. The pinned `ScaleFactorMem` counters wrap after a complete
configured loop; rs1 bit 62 resets requantizer counters but is not wired to
the scale read counters. Correct command scheduling and K-wave arithmetic
still require matching RTL simulator evidence. No larger contraction is
claimed executable by this prototype.

The packer refuses payloads above one active window. A compiler must
schedule additional uploads and choose scale banks for larger contractions;
that scheduling and all MX arithmetic lowering remain open.
The selected RTL's `MX_LOAD_LUT` DMA reads through the next 8-byte boundary,
so callers must use the padded result as the actual source buffer.

Run `python -m pytest tests/test_layout.py -q` from this directory.
