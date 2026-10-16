# MX Gemmini MLIR

This repository provides an out-of-tree MLIR dialect and target compiler
components for [Gemmini MX](https://github.com/ucb-bar/gemmini). It models MX
matrix commands, precision and resource contracts, VPU operations, and the
MX-to-Muon handoff. The target package also integrates with
[Merlin](https://github.com/ucb-bar/merlin) to compile complete captured model
graphs while Merlin owns the portable host path and session state.

The code is licensed under [Apache-2.0](LICENSE).

## Compile a model session

The [end-to-end model guide](docs/compile_e2e_models.md) gives commands to
capture, compile, link, and run TinyLlama, DeepSeek-R1-Distill-Qwen-1.5B,
Gemma 2 2B, and SmolVLA with model2MLIR, Merlin, and the selected MX+VPU
profile. It uses FP4 for eligible ordinary matmuls and FP8 for eligible
attention contractions. Unmapped operations execute in Merlin's compiled
host path.

All four pretrained model sessions have linked as standalone RV64 Spike
executables. The [session evidence](docs/evidence/four_vpu_session_links_20261012/README.md)
records their stage coverage and artifact identities. One
[full-checkpoint TinyLlama decode stage](docs/evidence/tiny_vpu_decode_spike_20261012/README.md)
has completed on the pinned MX+VPU Spike extension. Full VPU-enabled sessions,
independent quantized numerical comparisons, and performance parity remain
under qualification. The selected contract is marked
`structural_unqualified` and `phase0_admitted: false`.

## Build and inspect the dialect

The C++ dialect defines and verifies typed MX operations. With an installed
LLVM/MLIR toolchain and Ninja:

```sh
cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir
cmake --build build --target mx-gemmini-opt
build/tools/mx-gemmini-opt /path/to/input.mlir -o /path/to/verified.mlir
```

The Python target package provides profile checks, source-bound object
lowering, and Merlin catalog/VPU integration. Install it in the Python
environment used for compilation:

```sh
python -m pip install -e '.[ir,capture]'
```

`mx-gemmini-opt` verifies and transforms typed target IR. The full-model
entry point is the Merlin integration in `tools.build_merlin_stage` and
`tools.link_merlin_session`; the [model guide](docs/compile_e2e_models.md)
shows that path. The [quantization handoff](docs/quantization_handoff.md)
describes the dialect's typed contract and verifier.

## Repository map

| Location | Purpose |
| --- | --- |
| `include/MxGemmini/`, `lib/`, `tools/` | C++ dialect, verifiers, and command-line tools |
| `mx_gemmini_support/` | Target profiles, precision planning, physical lowering, VPU code generation, and Merlin adapter |
| `profiles/` | Configurations derived from selected Gemmini MX RTL revisions |
| `contracts/` | Merlin-facing target capability declarations |
| `docs/evidence/` | Captures, source comparisons, build receipts, and simulator results |

The [configuration guide](docs/current_mx_profiles.md) distinguishes Rocket,
standalone, VPU, precision, and mesh variants. The
[application-kernel coverage map](docs/nicolas_application_kernel_coverage.md)
tracks compiled equivalents of source kernels. The
[physical compiler record](docs/compiled_mx_pipeline.md) contains detailed
kernel-level lowering and Spike comparisons. The
[full-model status](docs/full_model_compilation.md) summarizes graph coverage,
session linking, and the remaining numerical gates.
The [connected-dispatch experiment](docs/evidence/connected_dispatch_20261016/README.md)
checks Merlin's shared residency planner through generated MX VPU binaries on
Spike, with identical captured PyTorch outputs and fewer intermediate transfers.

## Source-kernel regression tests

Tests that compare generated objects with the original MX source kernels
require external checkouts and a RISC-V toolchain. Set `NICOLAS_NPU_ROOT` to
the `npu-exploration` checkout, `NICOLAS_MX_RTL_ROOT` to the pinned Gemmini MX
RTL checkout, and `RISCV_ROOT` to the RISC-V toolchain before running
`python -m pytest tests`. Some tests also accept `MX_GEMMINI_RTL_ROOT` and
`MX_GEMMINI_RISCV_ROOT` for the same respective paths. Tests that need an
unavailable external input skip rather than assuming a machine-specific path.
