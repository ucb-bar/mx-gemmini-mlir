# MX Gemmini `2029218` diagnostic

This is a bounded RTL diagnostic for the unreviewed
[`2029218` software-contract candidate](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml).
The default Merlin provider remains pinned to `f016739`; an explicit
candidate selection can use TorchAO for operand-only capture. These results
do not admit a Phase 0 corpus or qualify full-model behavior.

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

The OOT source check resolves `defaultMxFPConfig` plus the standalone
overrides in the pinned `ConfigsFP.scala`. The selected fields are a 16×16
mesh, four scratchpad banks, two accumulator banks, 256 KiB scratchpad,
64 KiB accumulator, 32-element scale blocks, MX scaling enabled, and a
present LUT. The built-in nonlinear activation, normalization, and max-pool
switches are all false; the candidate software spec places those operation
families on the host. The check rejects an accelerator placement for a
disabled family. These are source-config facts, not evidence of host lowering
or complete elaboration closure.

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

## Explicit-load 64-square follow-up

The package's existing `canonical_bringup_source(format, "square64")`
generated one 64×64×64 contraction per format. Each used explicit contiguous
funct-27 scale loads, the generated header above, and the same simulator.
All three programs exited zero with zero BF16 mismatches.

| Format | C SHA-256 | ELF SHA-256 |
| --- | --- | --- |
| MXFP8 | `629b9080382d526d79e6c42f2133f4d05ab8814c00bdd9a5636e373d3f25efa7` | `49dd00ee70b7d3a80ac7757b1c191bf1e3dbdd7a783e70893ac8051908b77e0c` |
| MXFP6 | `dde663622142c4b9ef0c52c10ffd75182bef643d2d854e20fbfe9ced69876c5f` | `1fd23cd53bdd3b634e366ba8923feff36a5b5f30868e4dd36b73ed9faea991ea` |
| MXFP4 | `96d5d0e2ec4f4afb9458787b567eddde8440169bd40d2965db33fc99eaa74047` | `dc190950fbe8b26e28427c65734ff8fce2c064b60cdf777bdcf9daa3cbbe6b83` |

The [64-square runtime receipt](evidence/latest_rtl_2029218_square64.json)
records the source, ELF, simulator, and output hashes. These bounded runs
support the explicit-load path; they do not qualify other shapes or a general
compiler schedule.

## Two-wave follow-up

`canonical_bringup_source(format, "split32x64")` forces a 32×32×64
contraction into two 32-element K waves. The second wave reloads both scale
operands and accumulates into the first wave's BF16 result. The FP6 command
stream keeps its LUT resident across the split. All three formats exited zero
with zero BF16 mismatches on the same latest-RTL simulator.

| Format | C SHA-256 | ELF SHA-256 |
| --- | --- | --- |
| MXFP8 | `3244e190f3fc6d86c78ec16e81d313a34ac8a86478ba1b08d26c732eb138cd40` | `1c841e38262e8aaaa93b491a07fa2039077cebb6add6472bd30c509542485972` |
| MXFP6 | `fb85fc8538dce128ae6d106409359598ddcd49934b44e62f2b8bb17bcf9f78d0` | `a836dc4afd3d9acc8d7342a67b304a6a71d54c929a00c6a9ebf317c33a7570ed` |
| MXFP4 | `cea3ce74c060bb60c2499b6c9bcc392b9ed21ddffb9d56cd9f36b942bc44c868` | `310ffe3e6adc7524437cd3e495e6d850180424da030722ff458ba330a4d763d9` |

The [two-wave receipt](evidence/latest_rtl_2029218_two_wave.json) pins the
executed binaries and outputs. This proves only the specified split and
command order; it does not establish capacity-driven wave scheduling or
resident activation-scale reuse.

Repeated half reuse, the full widened DMA field range, configurations without
a LUT, and whole-model quantization and host transfers remain untested.
The candidate contract remains `unreviewed`.
