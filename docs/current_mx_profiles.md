# Current MX profiles and VPU command lowering

The checked-in `profiles/gemmini-mx-cleanup-266c593/` directory is projected
from Gemmini `gemmini-mx-cleanup` at
`266c593f2cb51d7e3fe83fc0317072b585ac3c52` and its MxGen submodule at
`dba3e7e706efa156d96efe595cd9935c825ea14d`. It covers all 41 named
Gemmini MX fragments and all 40 Chipyard MX wrappers in that revision. Each
JSON profile includes the exact
Scala source hashes, array geometry, DMA and queue limits, scale RAM and LUT
configuration, requantizer lanes, VPU options, and legal operand format,
projection, and PE mode tuples. `MxAllAsymGemminiRocketConfig` has 25
format pairs and 36 legal tuple variants. Chipyard profiles also record
explicit Rocket, system-bus, L2, and serial-TL overrides. The VPU+FP4 wrapper
selects four L2 banks and a 128-cycle outer latency setting.

The source parser is a conservative projection of these Scala declarations,
not a Scala elaboration. `load_profile(..., rtl_root=...)` regenerates and
compares the whole profile to reject source drift or edited capabilities.
Every exported profile has `qualification: structural_unqualified`. It does
not certify numerical parity, command scheduling, a bitstream, or throughput.
Separate [Spike qualification receipts](compiled_mx_pipeline.md#dim16-all-asymmetric-source-matrix)
now cover 30 of 36 legal BF16 source modes in the DIM16 all-asymmetric Rocket
profile. The DIM8 and DIM32 checked-in source sets cover 21 of 36 modes each;
generated-mode tests from Nicolas's pinned model bring each mesh to **35 of 36**
BF16 modes passing on unmodified Spike, with independent generated-mode
reproductions. See the [mesh qualification index](compiled_mx_pipeline.md#dim8-and-dim32-all-asymmetric-source-matrices).
The Radiance direct E4M3×E4M3 source driver separately qualifies a 31st
distinct DIM16 mode with two full-output Spike runs; it is not part of
Nicolas's checked-in source-mode matrix.
The [plain MX Rocket profile](compiled_mx_pipeline.md#nicolass-plain-mx-rocket-profile-across-fp8-fp4-and-fp6)
now has direct compiler-on-Spike evidence for its three selected FP8, FP4,
and FP6 Radiance source kernels: 36,864 BF16 outputs match across two builds
and a fresh-source rerun. This profile has no VPU; the separate VPU profiles
remain the targets for VPU compositions.
The [selected-profile matrix](compiled_mx_pipeline.md#selected-legal-modes-in-six-additional-mx-rocket-profiles)
adds direct Spike evidence for 12 legal Radiance mode/shape pairs across six
more Rocket profiles. Each selected case matches its full source BF16 golden
twice; untested legal modes and Chipyard wrappers retain structural status.
The [DIM8/DIM32 Radiance cases](compiled_mx_pipeline.md#radiance-fp8-fp4-and-fp6-on-dim8-and-dim32)
also compile from the archived model2MLIR captures. Their BF16 references
are derived from Radiance's pinned host model for each target mesh, because
the checked-in source golden uses the DIM16 accumulator schedule. Six
selected FP8/FP4/FP6 cases match all **73,728** target mesh reference outputs per
run on the corresponding Nicolas Spike extensions.
Four more cells have separate full-output tests from Nicolas's validated
`gen_asym.py` data model, giving **35 of 36 distinct DIM16 modes** with
numerical parity on the unmodified Spike extension. The remaining direct
E4M3 × E4M3-LUT cell fails on that extension because its weight-lane packing
ignores a loaded E4M3 weight LUT; a local diagnostic patch matches the golden
but is not a qualified upstream model or FPGA result. See
[generated DIM16 probes](compiled_mx_pipeline.md#generated-dim16-mode-probes).
The corresponding DIM8/DIM32 direct-E4M3 × E4M3-LUT cells fail on the same
stock Spike branch and pass with the same isolated model correction; they
remain unqualified on the stock model.

Nicolas's public branch includes `MxE4M3VpuGemminiRocketConfig` and
`MxE4M3Fp4VpuGemminiRocketConfig`. Both build two 8-lane BF16 scratchpad
VPUs with `EXPSUB` and `EXPSUM`, plus `SPAD_REQUANT`. The latter config also
builds FP4×FP4 mode 0 beside direct E4M3 mode 8. The dialect exposes
`mx_gemmini.vpu_execute` and `mx_gemmini.spad_requant`; the physical command
lowerer checks profile gates, row bounds, and the RTL's funct 33/34 bitfields
before emitting a Rocket RoCC C issuer.
The [compiler-issued VPU evidence](compiled_mx_pipeline.md#compiler-issued-base-vpu-operations)
now covers the 12 base opcodes on Nicolas's pinned Spike model; the two
fused opcodes have a separate [source-bound qualification](compiled_mx_pipeline.md#compiler-issued-fused-expsub-and-expsum).
Together with the [variant qualifications](compiled_mx_pipeline.md#compiler-issued-vpu-broadcast-and-same-bank-variants)
and [ordering qualifications](compiled_mx_pipeline.md#compiler-issued-vpu-dependencies-and-memory-ordering),
they map all 29 named `vpu_ops.c` source checks to compiler-issued Spike
comparisons: all 14 opcodes, broadcast, same-bank placement, reduction length
one, chained VPU operations, DMA write-after-read, and a cross-VPU dependency.
These are independent source-bound programs rather than one general scheduler
for the entire source benchmark.
`spad_requant` may bind its E8M0 output destination as
`scale_buffer = "name"` with `scale_dram_address = 0`. The issuer then takes `name` as a
runtime pointer, checks that it fits the RTL's 33-bit address field, shifts it
into funct 34, and keeps the source/destination scratchpad fields separate.
The fixed-address form remains available for known baremetal mappings.
The source-bound compiler has also qualified a full
MM1→VPU×2→SPAD_REQUANT→resident MM2 command chain on Nicolas's Spike.
Its checked frontend and target inputs now form one SSA-connected typed chain
for the qualified 64³ E4M3 source case; general scheduling remains pending.

A separate source-bound BF16 matrix lowering now compiles FP8 and FP4 contractions with the MX+VPU profile to
standalone RV64 ELFs. FP6 needs its separate LUT profile; no current VPU
profile contains FP6 E3M2 compute. See [compiled source parity](compiled_mx_pipeline.md).
The source-bound compiler also orders one in-place BF16 VPU epilogue after
the matrix K waves; FP8 and FP4 ×2 runs match their exact derived goldens on
Spike. A 64×64×128 FP8 program composes the matrix, VPU×2, and tiled resident
SPAD_REQUANT; a separate 64×64×64 source-bound program also includes the
following resident matrix stage. Both have numerical parity on Spike.
A [four-output-tile FP8 case](compiled_mx_pipeline.md#four-tile-fp8-gemm-with-tilewise-vpu-epilogue)
connects contraction, VPU, and BF16 readout with typed SSA, repeats the
in-place VPU ×2 after each tile's final K wave, and matches all 65,536
source-derived BF16 values on pinned Spike. This uses the selected DIM16
MX+VPU profile and an explicit tilewise policy.
A [four-output-tile FP4 case](compiled_mx_pipeline.md#generated-four-tile-fp4-gemm-with-tilewise-vpu-epilogue)
uses the same typed path and checks all 65,536 BF16 values on Spike with the
FP4+VPU profile. Its 256×256 driver and header are explicitly generated from
pinned Radiance sources because no matching driver is committed upstream.
The [executed source-tile check](compiled_mx_pipeline.md#executed-generated-fp4-source-tiles-on-cyclotron)
confirms four separate Muon/MX tiles against the same generated golden on
Cyclotron. The naive 256×256 source driver itself does not implement four
output tiles, so this is a derived source decomposition.
The [captured scalar ADDS case](compiled_mx_pipeline.md#captured-bf16-scalar-adds-across-four-mx-output-tiles)
binds `matmul + 1.5` to one VPU command per output tile. FP8 and generated FP4
each match all 65,536 BF16 outputs on Nicolas's pinned Spike in two builds.
The FP8 general source CLI independently reproduces the captured program.
The [ordered affine case](compiled_mx_pipeline.md#ordered-scalar-mx-vpu-chain-from-a-pytorch-capture)
keeps each BF16 output tile resident for captured `* 2.0` then `+ 1.5`.
Both FP8 and generated FP4 match all 65,536 outputs on pinned Spike twice;
the general FP8 source CLI independently reproduces the same eight VPU
commands. Intermediate BF16 rounding is part of this execution contract.
General scheduling remains open. Selected FP8+VPU and FP4+VPU physical
programs issue through Muon MMIO and each match all 65,536 BF16 outputs on
an isolated patched Cyclotron functional model in two independent builds.
The FP4 program uses a generated Radiance fixture and tile-major output.
The stock model lacks the compiler DMA/VPU command subset. See
[Muon MMIO qualification](compiled_mx_pipeline.md#source-bound-muon-kernel-and-simulator-gap).

The [full BF16 source roster](compiled_mx_pipeline.md#dim8dim32-source-roster-on-nicolass-spike)
now runs from the pinned model2MLIR capture through compiler-generated
RV64 ELFs on Nicolas's DIM8 and DIM32 Spike extensions. Each mesh matches
all 23 FP4/FP6/FP8 drivers and 376,832 target-model BF16 outputs in two
runs. The target reference records the geometry-specific accumulator schedule
and Nicolas's product floor. The separate VPU-enabled Rocket profile matches
the selected FP8 and FP4 matrix drivers twice; its vector instruction
coverage is documented above. The eight quantized-output source drivers are
qualified separately on DIM8, DIM16, and DIM32 with the generated Radiance
header requantization epilogue: each profile matches 90,112 output bytes and
3,328 E8M0 scales in two Spike runs. See the
[requant evidence](evidence/radiance_mx_requant_mesh_roster_266c593/index.json)
for the numerical reference boundary.
The VPU-enabled `MxE4M3Fp4VpuGemminiRocketConfig` also runs the complete
legal FP8/FP4 source GEMM subset: 18 BF16 drivers and six requant drivers,
each in two direct Spike runs. The
[profile archive](evidence/radiance_mx_vpu_legal_roster_266c593/index.json)
records all outputs and the compiler's FP6 rejection. These source drivers
do not issue VPU instructions; the vector qualifications above do.

No image-specific profile is checked in for the current VPU build. An
image-specific profile must bind the bitstream and elaborated Radiance config
before the compiler issues these commands through Muon MMIO. The physical
issuer has both Rocket and Muon transports, but the VPU MLIR lowerer currently
selects Rocket only.

Generate the profiles from the matching RTL checkout:

```sh
python -m tools.export_profiles --gemmini /path/to/gemmini --out /new/profile-directory
```

Bind an existing model2MLIR MX handoff to one selected profile. This keeps
the captured symmetric operand formats and projection choices. It will not
infer asymmetric quantization or switch direct E4M3 values to LUT indices:

```sh
python -m mx_gemmini_support.bind_profile \
  --mlir docs/evidence/model2mlir_radiance_mx_gemm_20261006.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini --out /tmp/mx-bound.mlir
python -m mx_gemmini_support.verify_profile_ir \
  --mlir /tmp/mx-bound.mlir \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --rtl-root /path/to/gemmini --mx-opt build/tools/mx-gemmini-opt
```

For an independently authored physical VPU/SPAD_REQUANT MLIR command module,
`python -m mx_gemmini_support.vector_lowering --mlir ... --profile ...
--rtl-root ... --mx-opt build/tools/mx-gemmini-opt --out ...` emits a bounded
Rocket C issuer. The original source-bound frontend test suite passed against
model2MLIR `7485a829c0195af0ec42820837d609e62e466564`; the current VPU
op qualification above uses model2MLIR
`e9ded36eb85abf2d9097ac4dc11457c825853388` and executes the emitted
programs on Spike. RTL simulator parity and cycle qualification for the
latest profiles are still required.
