# MX Gemmini `2029218` diagnostic

This is a bounded RTL diagnostic for the unreviewed
[`2029218` software-contract candidate](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml).
The selected TorchAO contract remains pinned to `f016739`. These results do
not admit a Phase 0 corpus or qualify full-model behavior.

## Build identity

An isolated Chipyard checkout at `e02074414158c85059c1dcb231ff2f8941254b9a`
elaborated `gemmini.GemminiMxFPStandaloneConfig,chipyard.RocketConfig` with
Gemmini `2029218197f771ce71416f859d975bea47b7aabc` and MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`. Both source trees were clean.
The committed Chipyard Gemmini gitlink is older: this diagnostic used an
explicit detached Gemmini checkout and recorded the actual revisions. Its
99-file Gemmini/MxGen Scala source census has SHA-256
`b16352e8f9d78b3f689154f4403b8b416fdab4394540ef7965f22d37ffc8a627`.
No Chipyard tooling or RTL source was edited.

JDK 17, firtool 1.75.0, and Verilator 5.022 produced the following artifacts:

| Artifact | SHA-256 |
| --- | --- |
| FIRRTL | `3f2d3f6e1aeaecef5c3a29c24cbe6ab10c0073331eb18186128c1a430de8920e` |
| Verilator executable | `1d9130ad0263fe640c6fe3d62b1dd2a3333aa6dde23e3ba8849db52211ec52f2` |
| Generated `gemmini_params.h` | `f4d98ad4b2a75674366aacb16dd5e87d4c68184f78f0c143704f73b65cc1bd60` |

Elaboration exited zero with 49 Chisel warnings. The build used the pinned
Boot ROM image and DRAMSim2 `44322e2f935d7dac83b7adf8dd270b41a54c6acb`.
The C programs used `gemmini-rocc-tests`
`df49519cda76867da5f01864c600ca7946dfb95d` and its pinned nested
submodules.

## Executed checks

Each 32×32×32 program loaded 32 logical E8M0 scale bytes per operand as two
16-byte rows separated by a 32-byte source pitch. The skipped bytes were zero,
so a load that ignored pitch would change the output. The alternate-half case
also used destination byte offset 4096, gated loads, selected that half for
both operands, and set the `CONFIG_SCALE_MEM` landed and managed-ready bits.
All four programs exited zero on the executable above and printed zero BF16
mismatches.

| Format | Variant | C SHA-256 | ELF SHA-256 |
| --- | --- | --- | --- |
| MXFP8 | `pitched` | `b9e88dc7f2ddbbf6c1ae163f6054d3791e9a6b2f5a13e6b7f630c15f9f2ecb54` | `2c4e77ea33fc40631dd49530cf7ad0c7974157900532dafd99f73362d0a3c058` |
| MXFP6 | `pitched` | `5c34cd7e95c0fbd20b1e92cc5544bee36d974e60070978a33f2577f164edd200` | `2a21ec2491a861d33daa11415d48ba2a337e587155a82abcb3bb9e9a48d99622` |
| MXFP4 | `pitched` | `4cdd1b818ddf72d03e65ea6802d360a1a5573cf69176d6fd3c8a5b7a32a95319` | `1339c45ffadcc9b5c9cff67aa11b306b08844a88966306510a90f954f73390ed` |
| MXFP8 | `pitched-gated-alt-half` | `7c59fa0798a1f60eab7260616c347a9f5539ed1e09ea4edf4ef0f97c45238a13` | `304ea66069d4dd7c790ddd458495f75f4c3eea0e363b688083434873ee9197e0` |

Generate the exact C source from this out-of-tree package with, for example:

```sh
python -m mx_gemmini_support.candidate_bringup \
  --format mxfp8 --variant pitched --output /artifact/mxfp8_pitched_scale_load.c
```

Compile it against the pinned test header with `MX_ROCKET`. From a directory
containing the pinned `dramsim2_ini` files, run:

```sh
<simulator> <ELF> +dramsim +max-cycles=500000 +loadmem=<ELF>
```

The ELF must be the first simulator argument. The generator refuses to
overwrite an existing source file and prints its digest.

## Loop-managed follow-up

The same simulator ran loops configured with funct 31/32. The FP8 64-square
case placed two 64-byte logical scale rows 128 bytes apart in source memory.
The FP6/FP4 cases deliberately used the same exact packed payloads that passed
above with explicit funct-27 loads.

| Format | Loop case | C SHA-256 | ELF SHA-256 | Result |
| --- | --- | --- | --- | --- |
| MXFP8 | 32×32×32, one row | `6264b4957cde674ec9f8eeec98eb62ed308689bd1ed9d5426676f4a6695de651` | `ca446f48af30bbdd8f14be58b0be8bebbc12097eb941e6493096d1c7630ad24a` | 0 mismatches, exit 0 |
| MXFP8 | 64×64×64, two pitched rows | `610fa4d1ac8778fa3d3f7ea6ab39bd0d5db5fabd01bf1525426c60d8db4143b7` | `7e3f092642ec681a50f3970a722b0a9c89d4d3e764dce1579fac2ff65d341394` | 0 mismatches, exit 0 |
| MXFP6 | 32×32×32, one row | `b9aa22f2788b49fb2119a0cb7a50b40bff90d42b4d527f15375c6650e884bd19` | `3e32f917847f6e6369a1eb82f138072948e8263812613cf06ced7a5c1bacebf6` | 768 mismatches, exit 255 |
| MXFP4 | 32×32×32, one row | `435f1fd234f436544a365ff5330a9bee0e5de7f16c8d72d2b8bacfc76d7546ae` | `87d95279e6ba5e00cf65e7ffe2493d464728d2ec7dc078ffa0179b09e183eea7` | 768 mismatches, exit 255 |

The [loop runtime receipt](evidence/latest_rtl_2029218_loop.json) records the
simulator, source, ELF, output hashes, and exact exit results for this rerun.

The failure pattern leaves 256 of 1,024 BF16 outputs correct. In
`LoopMatmul.scala`, `lds_row_bytes` is `max_i/max_j * block_size`; with this
configuration, it emits 16 bytes for the FP6/FP4 test's physical 32-byte scale
row. The underfilled row is a source-and-output inference, not a waveform
proof. The candidate contract therefore permits loop-managed scales only for
MXFP8 and requires explicit funct-27 scale loads for MXFP6/FP4. No RTL source
was changed.

The generator also reproduces these command streams with `--variant
loop-managed` or, for the tested MXFP8 64-square case, `--variant
loop-managed-pitched-64`. Its loop command IDs are emitted outside the pinned
software header, which does not yet define funct 31/32.

Repeated half reuse, the changed DMA column sizing, conditional LUT
construction, and whole-model quantization and host transfers remain untested.
The candidate contract remains `unreviewed`.
