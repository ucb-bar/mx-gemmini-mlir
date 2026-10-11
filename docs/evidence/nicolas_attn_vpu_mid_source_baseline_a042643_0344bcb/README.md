# Nicolas dense attention mid-shape source oracle

This archive pins the unmodified `attn_vpu_mid.c` wrapper and shared
`attn_vpu.c` from `npu-exploration` `0344bcb`. The source generator produced
`Sq=64`, `Sk=128`, `d=64`, seed 0 E4M3 operands and E8M0 scales. The
[compressed header](attn_vpu_mid.h.gz) includes its bit-exact mesh score
golden. The [expectation header](attn_vpu_mid_expect.h) records the four
stage hashes discovered on Nicolas's pinned Spike model.

The [unmodified source ELF](source.elf) ran twice on that model with identical
logs. The [archived log](source_spike.log) reports zero differing score words
out of 2,048 four-element words, matching score, probability, requantized
probability, and PV hashes, and `attn_vpu PASSED`.

A PyTorch `bmm(Q, Kᵀ) → scale by 1/8 → softmax → bmm(P, V)` module was
captured through clean local model2MLIR commit `a042643` using the FX
importer. Its [MLIR](attention.model2mlir.mlir) has no opaque operation;
the [worklist](attention.worklist.json) identifies QK as 64×128×64 and PV
as 64×64×128. The current MX compiler does not yet lower this complete
mid-shape graph. This archive provides its source and frontend oracle.

The [receipt](receipt.json) pins source, generator, model2MLIR, profile,
toolchain, Spike extension, ELF, graph, and stage hashes. The local
model2MLIR checkout had unrelated uncommitted changes, so the capture was
repeated from a clean `git archive` of `a042643`; both produced byte-identical
MLIR. No FPGA execution was used.
