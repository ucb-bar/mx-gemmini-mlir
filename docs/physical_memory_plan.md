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

The captured small-Llama adapter in npu-exploration qualifies the DIM16
FP8 E4M3 `matmul_6 → view_19 → sigmoid_0` edge. Its matrix summary uses
`MxGemmPlan` and checks the generated C's A/B move rows, scale selectors,
loop geometry, and readout row. It conservatively reserves the output's
`smem_rows` in the accumulator capacity. The actual emitted VPU `Command` and
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
