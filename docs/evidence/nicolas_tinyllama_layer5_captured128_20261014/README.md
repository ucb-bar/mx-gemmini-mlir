<a id="nicolass-tinyllama-layer-on-captured-inputs"></a>

# Reference TinyLlama layer on captured inputs

This is a **source-only section result**, not a compiler-versus-source
comparison. It runs the reference implementation’s fused MX+VPU layer-5 kernel from
`npu-exploration` `0344bcb` on the same 128 input token IDs used by the
model2MLIR TinyLlama prefill capture. The first 64 tokens populate its KV
cache; the kernel processes the remaining 64. The layer's nine source weights
are byte-identical to the corresponding captured checkpoint tensors
([weight check](weight_match.json)).

| Check | Result |
| --- | ---: |
| Source layer, Spike `rdcycle` from launch through `h_out` | 582,204 |
| Output stage hashes printed | 12 |
| `h_out` relative Frobenius error against PyTorch layer output | 7.1011% |
| PyTorch float64 layer reference error against PyTorch layer output | 0.0000207% |

The [Spike log](spike.log) contains phase timings, output hashes, and the
source's numerical checks. The [receipt](receipt.json) binds the result to
the captured input, source files, generated blob, ELF, toolchain extension,
catalog, and selected profile. The generator's
[small patch](capture_input.patch) replaces its WikiText input with the
captured token IDs. The [reproduction script](reproduce.sh) applies that
patch, generates data, builds the source ELF, and runs Spike.

The source kernel uses FP8 E4M3 for all MX projections and 64 query tokens.
The compiler's captured prefill uses **FP4 for the seven layer-5 projections**,
FP8 for its two attention contractions, and 128 query tokens. It has no
equal-work isolated layer timing or output match for this source program.
The 582,204 count must therefore stay out of the paired cycle table. It
cannot be multiplied by 22 to estimate a full-model result.

The source prints a mesh-ideal estimate of 11,141,120, above its 582,204
Spike `rdcycle` count. Its derived utilization exceeds 100%. That is a clear
signal that this simulator counter is **not FPGA latency** or a hardware
utilization measurement. It can support matched Spike comparisons only when
both programs run the same work under the same timing model.

## Reproduce

Use `npu-exploration` `0344bcba70b899039ccd0ac0801ca5605d46d3eb`,
its `microscaling-quant` submodule at
`587456ad53240b24e5adb18f1179973d572fa57b`, and Gemmini RTL
`gemmini-mx-cleanup` `266c593` with its `gemmini-rocc-tests` source.
The NPU generator expects `software/gemmini-rocc-tests` beside its checkout.
Set the paths below to those checkouts and to the captured prefill
`inputs.npz` from [the full-model guide](../../../compile_e2e_models.md).
Keep `WORK_DIR` in a task temporary directory; it holds a 50 MB data blob
and ELF.

```sh
export NPU_ROOT=/path/to/npu-exploration
export ROCC_ROOT=/path/to/software/gemmini-rocc-tests
export RISCV_ROOT=/path/to/riscv-tools
export RISCV_TEST_ENV=/path/to/riscv-tests/env
export SPIKE_EXTLIB=/path/to/libgemmini.so
export INPUT_IDS=/path/to/tiny_llama_session/stages/prefill/inputs.npz
export WORK_DIR="$TMPDIR/nicolas-tiny-layer5"
export PYTHON_BIN=/path/to/python-with-torch-transformers-and-mxq
bash docs/evidence/nicolas_tinyllama_layer5_captured128_20261014/reproduce.sh
```

The next comparable experiment must use the same 128-token causal work,
FP4 projection policy, FP8 attention policy, state handoff, and output
checker in both the compiler and source programs. A full-model source
baseline also needs the 32 one-token decode steps and all other model
operators. The [baseline requirements](../../stitched_model_source_baseline.md)
track those interfaces for TinyLlama and SmolVLA.
