# MX Gemmini MLIR

This repository provides an out-of-tree MLIR dialect and target compiler
components for [Gemmini MX](https://github.com/ucb-bar/gemmini). It models MX
matrix commands, precision and resource contracts, VPU operations, and the
MX-to-Muon handoff. The target package also integrates with
[Merlin](https://github.com/ucb-bar/merlin) to compile complete captured model
graphs while Merlin owns the portable host path and session state.

The code is licensed under [Apache-2.0](LICENSE).

## Compile a model session

The [model guide](docs/compile_e2e_models.md) gives pinned checkpoint and
repository versions, exact quantization/placement policies, and commands to
compile, link, run, check outputs, and package a FireSim workload.

Completed Rocket-Spike results use FP8 for every offloaded matrix binding:

| Workload | Modeled cycles | Comparison |
| --- | ---: | --- |
| TinyLlama, 160-token full forward | 10,326,818,367 | 1.236× versus an independent quantized program |
| TinyLlama, prefill + 32 cached decodes | 20,262,623,973 | 3.655× versus the corrected compiler baseline |
| SmolVLA, prefix + ten flows + action | 73,577,503,143 | 1.196× versus the previous compiler build |

The [comparison methodology](docs/fair_full_model_comparisons.md) explains
numerical gates, timing boundaries, source repositories, and kernel coverage.
The [Radiance Rocket port](docs/radiance_rocket_model_baseline.md) documents
the newer full Llama application and its completed numerical port checks;
its full-model measurements are separate from the table above.
The [artifact audit](docs/model_results_audit.md) checks the completed output
streams, actual executables, generated catalogs, and build receipts. FireSim
workload packaging is format-validated; FPGA execution remains unmeasured.
DeepSeek-R1-Distill-Qwen-1.5B and Gemma 2 2B have compilation/link evidence only.
The target contract remains `structural_unqualified`, with
`phase0_admitted: false`.

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
[target and layer-precision guide](docs/configure_target_and_precision.md)
shows how to select a hardware profile, identify a layer's compiler sites,
and build with a validated FP4/FP8 policy. The
[application-kernel coverage map](docs/nicolas_application_kernel_coverage.md)
tracks compiled equivalents of source kernels. The
[physical compiler record](docs/compiled_mx_pipeline.md) contains detailed
kernel-level lowering and Spike comparisons. The
[full-model status](docs/full_model_compilation.md) summarizes graph coverage,
session linking, and the remaining numerical gates.
The [connected-dispatch experiment](docs/evidence/connected_dispatch_20261016/README.md)
checks Merlin's shared residency planner through generated MX VPU binaries on
Spike, with identical captured PyTorch outputs and fewer intermediate transfers.
The [TinyLlama section experiment](docs/evidence/connected_model_section_20261016/README.md)
extends this path to two f32 model multiplies under the selected BF16 boundary
policy, with every output checked and host conversion overhead included.

## Source-kernel regression tests

Tests that compare generated objects with source kernels need the pinned
external repositories and a RISC-V toolchain. The source identities are in the
[release registry](examples/models/release.json); test-specific environment
variables are documented alongside each test. Run `python -m pytest tests`.
Checks requiring unavailable external inputs skip explicitly.
