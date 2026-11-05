# VPU softmax row layouts for selected MX lane counts

`emit_vpu_row_softmax` now accepts an explicit `lanes` argument for the
standalone row kernel. A non-DIM16 call must supply it. The selected Rocket MX
profile must have a VPU and a supported scratchpad row width of 4, 8, 16 or
32 bytes; a BF16 row holds `row_bytes / 2` values, so that number must equal
`lanes`. An actual build obtains the lane count with
`target_profile.source_vpu_lanes(profile, rtl_root)` after verifying the exact
Gemmini source checkout. Synthetic modified profiles in the tests establish
only code layout, not that a corresponding RTL build exists.

The full BF16 VPU softmax packs each logical row to a multiple of the selected
lanes, computes its reduction length from that padded width, and uses disjoint
scratchpad windows for scores, max, exponentials, sums and output. The hybrid
`f32_reduce` mode changes only its VPU exponential transfer width; its FP32
normalization remains on the host. Packed FP8 output and the striped16 FP32
reduction stay DIM16-only until their separate layout and numeric contracts
are qualified. The historical DIM16 generated C is byte-identical for both
the full BF16 and `f32_reduce` kernels.

The new tests cover 2, 4 and 16 BF16 lanes with DIM4, DIM8 and DIM32 row
widths, the existing eight-lane DIM16 source hashes, VPU address fields,
DMA byte widths, scratchpad capacity refusal and host C syntax. This is
structural emission evidence only. Model selectors and catalog lowering still
require DIM16, and no new lane configuration has Spike or RTL parity here.
