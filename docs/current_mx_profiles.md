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

Nicolas's public branch includes `MxE4M3VpuGemminiRocketConfig` and
`MxE4M3Fp4VpuGemminiRocketConfig`. Both build two 8-lane BF16 scratchpad
VPUs with `EXPSUB` and `EXPSUM`, plus `SPAD_REQUANT`. The latter config also
builds FP4×FP4 mode 0 beside direct E4M3 mode 8. The dialect exposes
`mx_gemmini.vpu_execute` and `mx_gemmini.spad_requant`; the physical command
lowerer checks profile gates, row bounds, and the RTL's funct 33/34 bitfields
before emitting a Rocket RoCC C issuer.
`spad_requant` may bind its E8M0 output destination as
`scale_buffer = "name"` with `scale_dram_address = 0`. The issuer then takes `name` as a
runtime pointer, checks that it fits the RTL's 33-bit address field, shifts it
into funct 34, and keeps the source/destination scratchpad fields separate.
The fixed-address form remains available for known baremetal mappings.
This pointer binding is command-level support; a compiler-generated numerical
VPU→requant→matmul chain is still pending.

A separate source-bound BF16 matrix lowering now compiles FP8 and FP4 contractions with the MX+VPU profile to
standalone RV64 ELFs. FP6 needs its separate LUT profile; no current VPU
profile contains FP6 E3M2 compute. See [compiled source parity](compiled_mx_pipeline.md).
The source-bound compiler also orders one in-place BF16 VPU epilogue after
the matrix K waves; FP8 and FP4 ×2 runs match their exact derived goldens on
Spike. A 64×64×128 FP8 program now composes the matrix, VPU×2, and tiled
resident SPAD_REQUANT with numerical parity on Spike. The following resident
matrix stage, general scheduling, and Radiance MMIO composition remain open.

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
Rocket C issuer. The source-bound current model2MLIR test suite passed against
model2MLIR `7485a829c0195af0ec42820837d609e62e466564`. This is a frontend
and command encoding check. RTL simulator parity and cycle qualification for
the latest profiles are still required.
