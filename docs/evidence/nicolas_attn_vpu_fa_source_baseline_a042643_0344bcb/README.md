# Nicolas dense attention flash-shape source oracle

This archive pins the unmodified `attn_vpu_fa.c` wrapper and shared
`attn_vpu.c` from `npu-exploration` `0344bcb`. The source generator produced
`Sq=64`, `Sk=256`, `d=128`, seed 0 E4M3 operands and E8M0 scales. The
[compressed header](attn_vpu_fa.h.gz) includes its bit-exact mesh score
golden. The [expectation header](attn_vpu_fa_expect.h) records the four
stage hashes discovered on Nicolas's pinned Spike model.

The [unmodified source ELF](source.elf) ran twice on that model with identical
logs. The [archived log](source_spike.log) reports zero differing score words
out of 4,096 four-element words, matching score, probability, requantized
probability, and PV hashes, and `attn_vpu PASSED`.

A PyTorch `bmm(Q, Kᵀ) → scale by 1/√128 → softmax → bmm(P, V)` module was
captured through clean local model2MLIR commit `a042643` using the FX
importer. Its [MLIR](attention.model2mlir.mlir) has no opaque operation;
the [worklist](attention.worklist.json) identifies QK as 64×256×128 and PV
as 64×128×256. The [receipt](receipt.json) pins source, generator,
model2MLIR, profile, toolchain, Spike extension, ELF, graph, and stage
hashes. The capture used a clean `git archive` of `a042643` because the
local model2MLIR checkout has unrelated uncommitted changes. No FPGA
execution was used.
