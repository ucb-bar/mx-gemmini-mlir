# Nicolas single-tile debug source audit

Run `python -m tools.audit_nicolas_single_tile_debug --rtl-root
/path/to/gemmini-mx-cleanup-266c593 --check` to verify the
[machine-readable audit](index.json) against Nicolas's pinned source.

`matmul_single_tile_test.c` is a debug program, not a numerical source oracle.
Its active MVIN loops start at tile index one, so for the header's 32×32×32
shape and 16×16×16 tiles, they load only the `(1,1)` operand tiles. It uploads
constant `0x7f` scales rather than the header scales, reads through fixed MMIO,
and has its BF16 comparison loop commented out. The PASS string follows an
`errors = 0` initialization that no active code changes. The header declares
16 identical K-scale groups even though a 32-element K dimension needs one.

This audit binds those observations to exact source and header hashes. It is
**not** a compiler output or Spike execution receipt. The compiler already
matches the 32×32×32 FP8 BF16 golden from the separate, checked
`matmul_tiled_fp8_32x32x32.c` source program; that does not establish parity
with this debug program's active MMIO sequence.
