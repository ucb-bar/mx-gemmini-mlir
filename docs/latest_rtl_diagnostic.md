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

Compile it against the pinned test header with `MX_ROCKET`, then run the ELF
on the recorded executable with `+dramsim`, `+max-cycles=500000`, and
`+loadmem=<ELF>`. The generator refuses to overwrite an existing source file
and prints its digest.

These checks do not execute loop-managed funct 31/32, repeated half reuse,
the changed DMA column sizing, conditional LUT construction, or whole-model
quantization and host transfers. The candidate contract remains `unreviewed`.
