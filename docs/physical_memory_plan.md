# MX physical row planning: first connected edge

`mx_gemmini_support.physical_memory_plan` is a pure analysis pass over an
ordered, typed graph. It follows Merlin's `DispatchProgram` buffer identity
and `arena_plan` last-use placement rule, with explicit alias roots and
completion fences for asynchronous accelerator commands. It does not rewrite
the command stream.

Each value declares a source, storage space, extent in physical rows (bytes
for host buffers), and optionally an emitter-fixed row or view alias. Each
event declares exact reads and writes. A typed `Command` is checked against
its encoded DMA or VPU row fields; an opaque matrix dispatch requires a
profile/plan/driver/effect certificate. The pass refuses unsupported command
effects, missing definitions, unknown capacities, overflow, live row overlap,
host access before async completion, and an async event with no matching
fence. A fence releases only the queues named by that event.

The source-bound v3 profile supplies scratchpad row width/capacity,
accumulator rows, and scale-memory line capacity. A and B scale selectors
receive separate halves of the selected scale memory. The profile does not
record the total LUT table row capacity, so the receipt reports LUT capacity
as unknown and any requested LUT allocation is refused.

The captured small-Llama adapter in npu-exploration qualifies the source-named
DIM16 FP8 E4M3 configurations with 64 KiB or 32 KiB accumulator for the
`matmul_6 → view_19 → sigmoid_0` edge. Its matrix summary uses
`MxGemmPlan` and checks the generated C's A/B move rows, scale selectors,
loop geometry, and readout row. The 64 KiB build retains the original
conservative `smem_rows` reservation, which sizes the output scratchpad
region rather than exact accumulator writes. For the 32 KiB build,
`source_bound_single_fp8_accumulator` pins Gemmini `12912323` and the hashes
of `LoopMatmul`, `MxGeom`, `ExecuteController`, and `AccumulatorMem`. The
selected single-throughput DIM16 loop has 1 output-row tile and 22
output-column tiles. `LoopMatmulExecute` packs four column tiles into each
16-row accumulator window; `LoopMatmulStCSpad` reads the same six windows.
Both bounds are 96 rows, below the source capacity of 256 rows. The helper
rejects another revision, changed RTL file, unsupported profile, or a
larger tile count that exceeds capacity. The actual emitted VPU `Command` and
`Fence` objects are shared with its C issuer and checked row by row. The
guest host materializes the BF16 mesh output as FP32; the two captured views
alias that host buffer. They never alias scratchpad or accumulator storage.

The machine-readable receipt records every value's source, physical placement
or alias, definition and completed last use, per-space peak live rows and high
water, event effects, command fields, and the exact profile, matrix plan,
matrix generator, VPU issuer, and final driver hashes. A high-water of the
whole scratchpad can coexist with a much smaller live peak because the
matrix weight tiles are fixed near the top of memory.

This first adapter certifies one connected edge. The generic pass accepts
the DIM4/8/16/32 source-legal row geometries, but a new matrix schedule needs
its own certified footprint and completion summary before the runner can use
the pass for that schedule. The receipt is a static safety gate followed by
selected Spike parity, not whole-model allocation or FireSim evidence.
