# Compile and run complete models

This guide builds captured models with model2MLIR, Merlin, and the MX Gemmini
backend. It covers the three workloads with completed, audited executions:
TinyLlama's 160-token forward, TinyLlama's 128-token prefill followed by 32
cached decodes, and SmolVLA's prefix encoder followed by ten flow steps and
action decoding. The result is a standalone RV64 executable, with a separate
packaging step for FireSim.

The published cycle counts are from the `e4m3_vpu` Rocket-Spike timing model
with `mem.host_tracking=0`. They are not FPGA measurements. See the
[results and comparison methodology](fair_full_model_comparisons.md) and the
[artifact audit](model_results_audit.md) before using a speedup in a report.

The newer [TinyLlama host-softmax recipe](../examples/models/optimization_policies/README.md#build-the-rne-host-softmax-alternative)
has separate implementation pins and complete build/comparison commands.
Its finite-maximum configuration passes the Radiance-relative numerical
gates at 6,997,758,443 modeled cycles. The fixed release and its repository
pins below remain the reference for the other published workload recipes.

## Models and inputs

| Workload name | Checkpoint | Executed work | Retained numerical reference |
| --- | --- | --- | --- |
| `tinyllama-forward` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | All 22 layers and LM head, 160 fixed token IDs; all 160 logit rows | Floating-point model, all 5,120,000 F32 logits; Radiance-derived Rocket port at all 160 positions, older source at 33 |
| `tinyllama-cached` | Same checkpoint | 128-token prefill, then 32 one-token decodes with persistent KV state | Floating-point trajectory; 33 last-token logit tensors and all output/state digests |
| `smolvla` | `lerobot/smolvla_base` | Prefix, ten recurrent flow steps, action decode | Floating-point trajectory preserving the model's native BF16 backbone; ten flow outputs and final action |

TinyLlama uses checkpoint revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. SmolVLA uses
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
The token stream is fixed and the SmolVLA observations are synthetic model-ready
inputs. SmolVLA receives one F32 image of shape `1×3×512×512`, one Boolean
image-valid flag, `1×48` I64 text IDs and a Boolean text mask, and `1×32` F32
state. Its action/noise shape is `1×50×32`. This workload does not include image
preprocessing, tokenization, a camera driver, or robot control.

DeepSeek-R1-Distill-Qwen-1.5B and Gemma 2 2B have also been captured and linked.
Their [link evidence](evidence/four_vpu_session_links_20261012/README.md) is a
compilation result; this release does not publish completed numerical or
performance results for them.

## Versions and environments

The machine-readable [release registry](../examples/models/release.json) pins
repository commits, checkpoint revisions, target profile, exact policies,
weight identities, measured executable hashes, and tool binary hashes. Its
MX revision identifies the compiler baseline; the reproduction helpers and
capture supplement are supplied by this checkout. Historical build receipts
remain authoritative for the producers of each measured artifact.

The Merlin revision below is retained under the tag
`mx-gemmini-measurements-20261019`. Use that revision for the historical
compiler baseline; later source cleanup does not replace the producer
revisions recorded in the evidence.

| Repository | Revision used by this release |
| --- | --- |
| [mx-gemmini-mlir](https://github.com/ucb-bar/mx-gemmini-mlir) | `fd5fff33aea54ee1e175b97bb88496351fda88e9` compiler baseline |
| [Merlin](https://github.com/ucb-bar/merlin) | `5e81a25d27246b490f947c9201fd20fabe27c69e` |
| [model2MLIR](https://github.com/ucb-bar/model2MLIR) | `f5e5c674a585bebbc2507dbd028ebf9d26e2afbc` |
| [Gemmini](https://github.com/ucb-bar/gemmini) | `266c593f2cb51d7e3fe83fc0317072b585ac3c52` |
| [mxgen](https://github.com/ucb-bar/mxgen) | `dba3e7e706efa156d96efe595cd9935c825ea14d` |
| [libgemmini](https://github.com/ucb-bar/libgemmini) | `876631072ecc7cc8197df5dc69c4d26bd4cd5f54` |
| [gemmini-rocc-tests](https://github.com/ucb-bar/gemmini-rocc-tests) | `350547f9843f46f485d4d1dd4c20b2f52f4844bf` |
| [LLVM/MLIR](https://github.com/llvm/llvm-project) | `a47bddccec30255619bb8c37fa59700e661d4e66` |
| [Chipyard](https://github.com/ucb-bar/chipyard) | `d45f86f4cca379715ac0ceb3a9f2369927796794`, toolchain/startup checkout |
| [FireSim](https://github.com/firesim/firesim) | `b084672c2f8cf32e55d78f73a001074c23f8a2b8`, workload-format validation |

The Chipyard and FireSim pins above do not identify a measured FPGA design.
FireSim hardware must include the pinned Gemmini integration described below.

Use Python 3.12. The capture environment used PyTorch 2.10.0, Transformers
5.3.0, LeRobot 0.5.1, NumPy 2.2.6, xDSL 0.65.0, safetensors 0.8.0, and
torchao 0.16.0, with model2MLIR's FX importer. The compiler environment used
PyTorch 2.10.0, Transformers 5.9.0, NumPy 2.4.6, xDSL 0.65.0, and
`torch-mlir==20260531.828`. Keep these environments separate. The original
wheel URLs and a complete environment lock were not retained; installing
these versions alone does not establish a bit-for-bit historical rebuild.

Install the checkouts in the corresponding environments:

```sh
export MX_ROOT=/path/to/mx-gemmini-mlir
export MERLIN_ROOT=/path/to/merlin
export M2M_ROOT=/path/to/model2MLIR
export CAPTURE_PY=/path/to/capture-env/bin/python
export COMPILER_PY=/path/to/compiler-env/bin/python
export LLVM_INSTALL=/path/to/llvm-mlir/install
export RISCV_ROOT=/path/to/riscv-tools
export TORCH_MLIR_WHEEL=/path/to/compatible-torch-mlir-wheel.whl
export ARTIFACT_ROOT=/path/to/model-artifacts
export PYTHONPATH="$MX_ROOT:$MERLIN_ROOT/src:$M2M_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MERLIN_COMPILER_PYTHON="$COMPILER_PY"
export MERLIN_MLIR_INSTALL="$LLVM_INSTALL"

"$CAPTURE_PY" -m pip install -c "$MX_ROOT/examples/models/capture-constraints.txt" \
  -e "$M2M_ROOT[quant,examples]" -e "$MERLIN_ROOT" -e "$MX_ROOT[ir]" \
  'lerobot[smolvla]==0.5.1'
"$COMPILER_PY" -m pip install -c "$MX_ROOT/examples/models/compiler-constraints.txt" \
  -e "$MERLIN_ROOT" -e "$MX_ROOT[ir]" 'torch==2.10.0' 'transformers==5.9.0'
# Supply the recorded torch-mlir build in the compiler environment:
"$COMPILER_PY" -m pip install --no-deps "$TORCH_MLIR_WHEEL"
mkdir -p "$ARTIFACT_ROOT"
cd "$MX_ROOT"
```

The constraints record the observed versions rather than a complete lock.
The `torch-mlir` wheel path is a required external input; use the compatible
Python 3.12 build from your toolchain distribution, or supply another
compatible torch-mlir build and qualify its outputs as a fresh toolchain.
Do not infer a wheel download URL from the version string.

The RV64 tools used GCC 13.2.0 (`gc891d8dc23e`) and Spike 1.1.1-dev.
Build LLVM with MLIR, Clang, and the RISC-V target enabled. Build the Spike
extension from the pinned Gemmini/libgemmini sources, following their build
instructions, and set `SPIKE_EXTLIB` to the resulting `libgemmini.so`.
The registry records the measured Clang, Spike, extension, and `libm.a`
SHA-256 values. The audited Spike source checkout is
`9c190a07c6838f6392bafa4ad83acea462c7f759`; the recorded executable hash is
the authority for matching the historical simulator binary. The host-math guard additionally checks the actual library;
a different library must be reviewed as a new numerical configuration.

## Quantization and placement

Capture with `--formats fp32`, then select MX formats in the backend. The
frontend's `fp8` option is a different weight-only transformation and does
not reproduce these runs. In SmolVLA, the `fp32` capture label leaves the
native BF16 backbone intact. Its external checkpoint contains 474 BF16 and
26 F32 tensors. TinyLlama's external capture contains 201 F32 tensors:
200 model checkpoint tensors plus a captured constant.

All offloaded matrix bindings in the completed runs use **MXFP8**, with E4M3
codes and E8M0 scales per group of 32. Packing rounds operands through BF16.
The VPU operates on BF16; logical inputs, outputs, and host operations retain
the types in the captured IR, often F32. Checkpoint prepacking moves constant
packing out of model execution; it does not remove matrix execution.

For each group, `amax` is the largest absolute BF16-rounded operand. The
shared scale exponent is `max(-23, floor(log2(amax)))`; an all-zero group
uses −23. The E8M0 byte stores that exponent plus 127. Each code is the
E4M3 rounding of the BF16-rounded value divided by the shared power of two.
Both rounding steps use nearest, ties to even. Groups run along the matrix
reduction axis: one group per 32 activation values in a row, or per 32
weight values in an output column. This scale rule is part of the measured
quantization policy, alongside the E4M3 and E8M0 encodings. The
[reference packer](../mx_gemmini_support/runtime_quant.c) defines it, and the
[learned-weight audit](evidence/radiance_rocket_optimization_20261019/source-compiler-weight-equivalence.json)
checks the encoded payloads against the Radiance-derived port.

| Stage | FP8 matrix bindings | Host matrix bindings | Other placement |
| --- | ---: | ---: | --- |
| TinyLlama forward / cached prefill | 199 | 1 | VPU reductions, normalization components, activations and selected adds; 22 host softmax sites and 22 host elementwise sites |
| TinyLlama decode | 199 | 1 | Same host softmax/elementwise counts; 45 adds remain on host, 88 on VPU |
| SmolVLA prefix | 243 | 0 | 28 host softmax and 52 host elementwise sites; selected remaining arithmetic on VPU |
| SmolVLA flow | 148 | 0 | 16 host softmax and one host elementwise site; selected remaining arithmetic on VPU |
| SmolVLA action decode | 0 | 0 | Compiled host code |

Counts describe captured binding sites, not dynamic matrix launches. The
[policy files](../examples/models/policies/) enumerate every selected operation
by identity and the registry records the full placement census. These files
are checked against the actual build probes by the audit tool. Sensitive
attention and normalization boundaries therefore cannot silently move to a
lower precision under a broad default policy. Mixed FP4/FP8 remains an
experimental alternative and is not the policy behind the current numbers.

For another target or mixed layer precision, follow
[configure the target and choose layer precision](configure_target_and_precision.md).
It covers source-checked profiles, layer-to-region mapping, policy validation,
and rebuilding custom stages separately from the fixed release recipe.

## Obtain captures and checkpoint files

For an exact input replay, use the archived captures. Exporting requires the
external `weights.safetensors` files because multi-gigabyte checkpoint data
is not distributed in this repository. Expected file hashes are in the
registry. TinyLlama prefill, decode, and forward use the same checkpoint file;
SmolVLA prefix and flow also share one file. Action decode needs none.

```sh
"$CAPTURE_PY" -m tools.model_release export --model tinyllama-cached \
  --weights "all=/path/to/tinyllama/weights.safetensors" \
  --out "$ARTIFACT_ROOT/tinyllama-cached-inputs"

"$CAPTURE_PY" -m tools.model_release export --model smolvla \
  --weights "all=/path/to/smolvla/weights.safetensors" \
  --out "$ARTIFACT_ROOT/smolvla-inputs"

"$CAPTURE_PY" -m tools.model_release export --model tinyllama-forward \
  --weights "all=/path/to/tinyllama/weights.safetensors" \
  --out "$ARTIFACT_ROOT/tinyllama-forward-inputs"
```

Each destination must be fresh. The exporter verifies archive members,
capture supplements, policies, and checkpoint hashes, and writes an
`inputs.json` checksum manifest. By default it copies checkpoint data;
`--link-weights` can share an immutable local file on the same filesystem.

To inspect the inputs before obtaining checkpoint files, replace `--weights`
with `--capture-only`. This exports reference inputs and policies but cannot
be compiled. Obtain the pinned Hugging Face snapshots using
`huggingface_hub.snapshot_download(repo_id=..., revision=...)`, then generate
externalized weights through the model2MLIR capture recipes below. The
TinyLlama loader at the pinned revision does not accept a checkpoint revision:
use an isolated Hugging Face cache containing only the pinned snapshot, then
set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. SmolVLA's loader explicitly
pins its revision. Its LeRobot constructor also loads the VLM backbone and
processor. Cache `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` before an offline fresh capture.
The [frontend dependency record](../examples/models/frontend-dependencies.json)
binds the metadata observed during this audit; the original capture did not
separately bind that initialization dependency. The final externalized
checkpoint hash is the required correspondence check. Never accept a changed
checkpoint merely because its model name matches.

For a fresh capture, first materialize the recorded source inputs without
checkpoint files:

```sh
"$CAPTURE_PY" -m tools.model_release export --model tinyllama-cached \
  --capture-only --out "$ARTIFACT_ROOT/recapture-tinyllama"
"$CAPTURE_PY" -m tools.model_release export --model smolvla \
  --capture-only --out "$ARTIFACT_ROOT/recapture-smolvla"

"$CAPTURE_PY" - <<'PYINPUTS'
import os
from pathlib import Path
import numpy as np
root = Path(os.environ["ARTIFACT_ROOT"])
tiny = root / "recapture-tinyllama/capture/stages"
with np.load(tiny / "prefill/inputs.npz") as p, np.load(tiny / "decode/session_inputs.npz") as d:
    ids = np.concatenate((p["in0"].reshape(-1), d["decode_tokens"].reshape(-1)))
np.save(root / "recorded-token-ids.npy", ids)
smol = root / "recapture-smolvla/capture/stages"
with np.load(smol / "prefix_encode/inputs.npz") as p, np.load(smol / "flow_denoise/inputs.npz") as f:
    np.savez(root / "recorded-observation.npz", image=p["in0"], image_mask=p["in1"],
             language_tokens=p["in2"], language_mask=p["in3"], state=p["in4"], noise=f["in2"])
PYINPUTS
```

The full-model capture recipes are:

```sh
"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" tiny_llama \
  --formats fp32 --python "$CAPTURE_PY" --output-root "$ARTIFACT_ROOT/recapture" \
  --env M2M_LLAMA_SESSION=e2e --env M2M_LLAMA_LAYERS= \
  --env M2M_LLAMA_CHECKPOINT_LAYERS= --env M2M_LLAMA_ATTENTION=eager \
  --env M2M_PREFILL_TOKENS=128 --env M2M_DECODE_TOKENS=32 \
  --env "M2M_LLAMA_TOKEN_IDS=$ARTIFACT_ROOT/recorded-token-ids.npy" \
  --env M2M_LLAMA_TOKEN_SOURCE=archived-model-inputs

"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" smolvla \
  --formats fp32 --python "$CAPTURE_PY" --output-root "$ARTIFACT_ROOT/recapture" \
  --env M2M_SMOLVLA_SESSION=e2e --env M2M_SMOLVLA_PRETRAINED=1 \
  --env M2M_SMOLVLA_VLM_LAYERS= --env M2M_SMOLVLA_EXPERT_LAYERS= \
  --env "M2M_SMOLVLA_INPUT_NPZ=$ARTIFACT_ROOT/recorded-observation.npz" \
  --env M2M_SMOLVLA_INPUT_SOURCE=archived-model-inputs
```

For a single 160-token forward, use the TinyLlama command with
`--env M2M_LLAMA_SESSION= --env M2M_SEQ=160` instead of the session selection.
The same token file covers all 160 positions. A fresh session capture writes `<model>/<model>_session/` with separate stage
bundles. Compare its externalized weight SHA-256 with the registry before
using that file with the archived capture. Serialization or frontend changes
can produce different bytes; that case requires a new capture/weight
correspondence check, not an override of the exporter. The archived input
replay is reproducible once the matching checkpoint and toolchain files are
available; a fully hermetic checkpoint-to-historical-ELF rebuild is not yet
provided.

## Compile, link, and run

Choose `MODEL=tinyllama-cached`, `smolvla`, or `tinyllama-forward`:

```sh
export MODEL=tinyllama-cached
export SPIKE_EXTLIB=/path/to/libgemmini.so

"$CAPTURE_PY" -m tools.model_release build --model "$MODEL" \
  --inputs "$ARTIFACT_ROOT/$MODEL-inputs" \
  --build-dir "$ARTIFACT_ROOT/$MODEL-build" \
  --merlin-root "$MERLIN_ROOT" --llvm-install "$LLVM_INSTALL" \
  --riscv-root "$RISCV_ROOT" --python "$CAPTURE_PY" \
  --spike-extlib "$SPIKE_EXTLIB" --plan "$ARTIFACT_ROOT/$MODEL-plan.json" \
  --execute
```

Omit `--execute` to review the complete command plan first. Build and plan
paths must be fresh. Omit `--spike-extlib` for compilation/linking without
execution. Full models can take hours in Spike. The helper selects the exact
per-stage policy and host-math guard, prepacked/pruned checkpoint objects,
BF16 transport, shared activation packing, the 2 GiB stage arena, and the
applicable RHS reuse/compaction options. It records the timing environment
in the plan. The standalone forward dumps all logits; session links dump all
33 selected TinyLlama logit tensors or all 11 SmolVLA action tensors.

Stage builds contain `model.elf`, lowered IR, the MX catalog, and
`mx_stage_probe.json`. Session builds additionally contain
`link/session.elf`, `link/mx_session_link.json`, `run/spike.log`, complete
`run/output_dumps/`, and `run/mx_session_run.json`. The linker verifies the
stage roster, object identities, shared weights, persistent-state bindings,
and arena/static-data placement. Structural admission is separate from
numerical qualification; a successful link alone is insufficient.

Check a completed session against every published output/state digest and
all complete output tensors:

```sh
"$CAPTURE_PY" -m tools.check_reproduced_model --model "$MODEL" \
  --link-receipt "$ARTIFACT_ROOT/$MODEL-build/link/mx_session_link.json" \
  --run-receipt "$ARTIFACT_ROOT/$MODEL-build/run/mx_session_run.json" \
  --out "$ARTIFACT_ROOT/$MODEL-output-check.json"
```

For `tinyllama-forward`, use
`--stage-build "$ARTIFACT_ROOT/tinyllama-forward-build/forward"` instead of
the two receipt arguments. Output parity preserves the published numerical
result; a performance ratio additionally requires equal tool hashes, profile,
timing settings, and measurement boundaries. A deadlocked timing model is
rejected even if the functional simulator exits zero and prints correct values.

Session totals include reset, program-state handoff, and existing step
intervals. Console output and phase bookkeeping are outside these intervals.
The forward interval covers token lookup through logits. Compilation,
checkpoint download, and constant prepacking are outside model execution.

## Package a session for FireSim

```sh
"$CAPTURE_PY" -m tools.package_firesim_model \
  --elf "$ARTIFACT_ROOT/$MODEL-build/link/session.elf" \
  --link-receipt "$ARTIFACT_ROOT/$MODEL-build/link/mx_session_link.json" \
  --name "mx-$MODEL" --out "$ARTIFACT_ROOT/firesim-$MODEL"
```

The packager checks the executable hash against its link receipt, selected
profile, RV64 ELF load segments and entry point. It creates a bare-metal
workload with no root filesystem:

```text
firesim-<model>/
  workloads/mx-<model>.json
  workloads/mx-<model>/session.elf
  requirements.json
  mx_session_link.json
  target_profile.json
```

Check the package with the pinned FireSim checkout before deployment:

```sh
"$CAPTURE_PY" -m tools.check_firesim_package \
  --package "$ARTIFACT_ROOT/firesim-$MODEL" --firesim-root /path/to/firesim \
  --out "$ARTIFACT_ROOT/$MODEL-firesim-check.json"
```

Copy the JSON and workload directory into FireSim's `deploy/workloads/`.
In the FireSim runtime configuration, set `workload.workload_name` to that
JSON filename and `target_config.default_hw_config` to your matching hardware
DB entry, then use the platform's normal `firesim infrasetup` and
`firesim runworkload` flow. The package format was tested with the pinned
[FireSim workload parser](https://github.com/firesim/firesim/blob/b084672c2f8cf32e55d78f73a001074c23f8a2b8/deploy/runtools/workload.py).

Hardware must implement `MxE4M3Fp4VpuGemminiRocketConfig` from the pinned
[Gemmini integration](https://github.com/ucb-bar/gemmini/blob/266c593f2cb51d7e3fe83fc0317072b585ac3c52/chipyard/GemminiConfigs.scala):
16×16 mesh, 256 KiB scratchpad, 64 KiB accumulator, 16 KiB scale memory,
32-element MX groups, VPU, and a 512-bit system bus. Use the full profile's
Chipyard overrides, not just the configuration name. `requirements.json`
records the load address range and runtime memory requirement. The audited
cached TinyLlama ELF is 2,439,169,264 bytes and requires an **8 GiB** DRAM
configuration after rounding the complete address span to a power of two.

No matching bitstream identifier or completed FPGA execution is qualified by
this release. In particular, the Spike executable uses host-target console
and exit support: verify that transport on the selected FireSim platform,
then retain the complete output/state checks after the hardware run. The
package status is deliberately `packaged_not_fpga_executed`. FPGA cycles
must be published as a new result rather than substituted for the Spike
model counts.
