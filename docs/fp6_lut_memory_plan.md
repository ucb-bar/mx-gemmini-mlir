# Captured FP6 LUT cache proof

The physical-memory planner certifies the three **direct** FP6 E3M2
codebook caches for the named DIM16 VPU configuration at Gemmini `12912323`.
The selected RTL inherits `GemminiLUTConfig(numEntries = [64,64,64],
numBits = [96,96,96])`. `MX_LOAD_LUT` selector 0 writes weight, selector 1
writes activation, and selector 2 writes output codebooks. This proof checks
the exact Gemmini revision and hashes of `ConfigsFP.scala`,
`MxConfigFragments.scala`, `Controller.scala`, and `QuantLut.scala`; a changed
source is refused until audited.

For captured `matmul_6`, the recipe chooses FP6 E3M2 with LUT group `G=3`
and the padded matrix is `[32,352,128]`. The emitter issues direct loads in
the order weight 44, activation 4, output 4, each 96-bit codebooks, then one
mesh dispatch and a completion fence. The output table is loaded even with
BF16 readout because the symmetric emitter emits selector 2; its lifetime is
reserved conservatively through the fence. The physical plan therefore
records cache peaks of 44/64, 4/64, and 4/64 codebooks.

The NPU adapter checks the generated C load selectors/counts/order, exact
provider array shape and hash, selected matrix plan, full driver hash, and
fence position before compiling the ELF. Its typed effects mark the three
cache writes and mesh reads asynchronous until the fence. Unknown LUT modes,
changed RTL, excess codebooks, overlapping live ranges, or an absent fence
are refused. The receipt is specifically a **LUT cache address/lifetime
proof**. It does not certify the mesh scratchpad, accumulator, scale memory,
the synthetic matrix-to-VPU bridge, or complete model inference.
