# Fresh checkout replay of the current MX compiler

This receipt comes from a new managed worktree at published compiler commit
`2b55be0`, with `mx-gemmini-opt` rebuilt from that worktree's source. The
[index](index.json) records the new binary SHA-256, compiler and external
source revisions, and the source checkout's unchanged Git status hash.

The read-only one-command replay captured all 31 Radiance MX GEMM drivers
through model2MLIR `e9ded36`, compiled the RV64 programs, ran them on the reference implementation’s
pinned Spike extension, and compared every output against the source golden.
It matched all 23 BF16-output and eight requantized-output cases. The
[reproduction receipt](reproduction.json), [frontend index](frontend/index.json),
and [Spike index](spike/index.json) match the archived baseline's source,
MLIR, ELF, and simulator-log digests for every driver. The source checkout was
Radiance `82be2c7`; `--compatible-source-revision` verified every driver and
header hash against the baseline `ee22e0b` roster. No FPGA was used.

The exact CLI is the [read-only roster command](../../compiled_mx_pipeline.md#latest-complete-radiance-mx-gemm-roster)
with `--read-only-source --compatible-source-revision --jobs 2`, pointing
`--mx-opt` to the newly built binary. This proof covers the MX GEMM source
roster, not full-model inference or cycle-level behavior.
