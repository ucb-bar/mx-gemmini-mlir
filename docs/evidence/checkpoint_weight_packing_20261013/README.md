# Checkpoint weight packing diagnostic

The selected MX+VPU profile routes fixed projections as FP4 and dynamic
attention products as FP8. An audit of the compiled model2MLIR stages found:

| Stage | Fixed checkpoint RHS, FP4 | Dynamic RHS, FP8 |
|---|---:|---:|
| TinyLlama prefill | 155 | 45 |
| TinyLlama decode | 155 | 44 |
| SmolVLA prefix | 186 | 57 |
| SmolVLA flow | 116 | 32 |

The audit traces each fixed RHS through the generated transpose and exact
BF16-to-F32 cast to a captured parameter. These are candidates for offline
packing when the executable is bound to that checkpoint. The full-model
catalog in these runs still repacks every RHS at execution time.

On the archived small-LLaMA `g` projection, the same RV64 Spike extension,
`e4m3_vpu` preset, `mem.host_tracking=0`, and Merlin native schedule produced:

| Checked adapter | `rdcycle` for one call | Logical BF16 mismatches |
|---|---:|---:|
| Live FP4 weight packing | 16,717,728 | 0 / 2,752 |
| Build-time FP4 weight packing | 1,269,283 | 0 / 2,752 |
| Fixed-weight cache, first call | 16,717,735 | 0 / 2,752 |
| Fixed-weight cache, second call | 1,266,389 | 0 / 2,752 |

The generated fixed codes and scales match the archived MX bundle byte for
byte. The two programs have the same issuer SHA256
`99f4e4c9a74e605a8e390088acedaa55132df7c0196c1e972240d98f690d45ff`
and schedule SHA256
`c96910ce018dc5306f7b9d5334ebe1931e13ba4dd1d5b7923719ba6870e270de`.
The live adapter spent 15,092,341 cycles packing the RHS and 1,230,204
packing the activation in a separate phase measurement. These are guest
`rdcycle` measurements for one small projection, not a whole-model speedup
claim or a calibrated FPGA latency.

An opt-in catalog build for the archived 19-site small-LLaMA source bound two
checkpoint parameter sites to separate cache entries while sharing six issuer
schedules across eight catalog kernels. A model-derived SmolVLA flow catalog
also compiled all 148 selected contractions with 116 fixed-weight cache
entries and 14 shared issuers. The complete TinyLlama and SmolVLA sessions
have not yet been rebuilt or timed with that cache option.

Reproduce the paired run with the commands in
[the model compilation guide](../../compile_e2e_models.md#6-inspect-fixed-weights-and-profile-conversion).
