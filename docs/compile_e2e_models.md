# Compile model sessions for MX Gemmini + VPU

This guide builds standalone RV64 session executables for TinyLlama,
DeepSeek-R1-Distill-Qwen-1.5B, Gemma 2 2B, and SmolVLA. The flow is
model2MLIR capture → Merlin whole-graph lowering → MX Gemmini matrix and VPU
lowering → session link. The resulting ELF contains compiled host code for
operations outside the selected MX/VPU patterns. No Python model is required
by the ELF at run time.

The commands below use the
[`MxE4M3Fp4VpuGemminiRocketConfig`](../profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json)
profile. The target selects FP4 for eligible ordinary matmuls and FP8 for
eligible attention and rotary contractions. Capture the source model in FP32:
the target applies this mixed-precision policy during compilation. The
model2MLIR `fp8` capture option represents a different frontend policy.

## Prerequisites

- Editable checkouts of this repository, [Merlin](https://github.com/ucb-bar/merlin),
  and [model2MLIR](https://github.com/ucb-bar/model2MLIR). Use compatible
  versions of all three; the [linked-session evidence](evidence/four_vpu_session_links_20261012/README.md)
  records the profile and artifact identities used for the published builds.
- A Python environment with the packages needed by model2MLIR's four loaders,
  including PyTorch, Transformers, LeRobot for SmolVLA, xDSL 0.65, NumPy,
  safetensors, and the MX package. Install each repository's dependencies in
  that environment. `--python` below points the capture driver to it.
- An installed LLVM/MLIR toolchain with `bin/clang`, an RV64 bare-metal
  toolchain with `riscv64-unknown-elf-gcc` and `spike`, and the Gemmini Spike
  extension built for the selected RTL profile. The extension is needed for
  execution, not for compilation or linking.
- Access to the four pretrained checkpoints. Set `HF_HOME` to a cache with
  those checkpoints if necessary. SmolVLA's loader requires its pinned
  `lerobot/smolvla_base` revision to be available locally.
- A fresh writable artifact directory with ample space. The linked Gemma ELF
  alone is approximately 13 GB; the other ELFs and intermediate objects add
  tens of GB. Choose a new directory for each run to keep its receipts and
  outputs together.

Set absolute paths from the MX repository root:

```sh
export MX_ROOT="$(pwd)"
export M2M_ROOT=/path/to/model2MLIR
export MERLIN_ROOT=/path/to/merlin
export LLVM_INSTALL=/path/to/llvm-install
export RISCV_ROOT=/path/to/riscv-tools
export CAPTURE_PY=/path/to/capture-environment/bin/python
export ARTIFACT_ROOT=/path/to/new-artifacts
export MX_PROFILE="$MX_ROOT/profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
export PYTHONPATH="$MX_ROOT:$MERLIN_ROOT/src:$M2M_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_ROOT/captures" "$ARTIFACT_ROOT/audits"
```

If the selected Python environment is new, install the three repositories and
the SmolVLA loader dependency. `uv` can install into that environment without
activating it:

```sh
uv pip install --python "$CAPTURE_PY" \
  -e "$MERLIN_ROOT" -e "$M2M_ROOT[quant,examples]" \
  -e "$MX_ROOT[ir,capture]" 'lerobot[smolvla]==0.5.1'
```

The variables identify toolchain installations and checkouts; they are not
hard-coded into the compiler. Keep the selected profile and Spike extension
from the same RTL configuration.

## 1. Capture the four sessions

Run these commands from any directory. The empty layer overrides clear the
small random-weight defaults in the model2MLIR capture recipes.

```sh
"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" tiny_llama \
  --formats fp32 --python "$CAPTURE_PY" \
  --output-root "$ARTIFACT_ROOT/captures" \
  --env M2M_LLAMA_SESSION=e2e --env M2M_LLAMA_LAYERS= \
  --env M2M_LLAMA_ATTENTION=eager

"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" deepseek_r1_qwen \
  --formats fp32 --python "$CAPTURE_PY" \
  --output-root "$ARTIFACT_ROOT/captures" \
  --env M2M_QWEN_SESSION=e2e --env M2M_QWEN_LAYERS= \
  --env M2M_QWEN_ATTENTION=eager

"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" gemma2_2b \
  --formats fp32 --python "$CAPTURE_PY" \
  --output-root "$ARTIFACT_ROOT/captures" \
  --env M2M_GEMMA_SESSION=e2e --env M2M_GEMMA_LAYERS= \
  --env M2M_GEMMA_SLICE_LAYERS= --env M2M_GEMMA_ENTRY=ids

"$CAPTURE_PY" "$M2M_ROOT/workloads/capture.py" smolvla \
  --formats fp32 --python "$CAPTURE_PY" \
  --output-root "$ARTIFACT_ROOT/captures" \
  --env M2M_SMOLVLA_SESSION=e2e --env M2M_SMOLVLA_PRETRAINED=1 \
  --env M2M_SMOLVLA_VLM_LAYERS= --env M2M_SMOLVLA_EXPERT_LAYERS=
```

Each result is under
`$ARTIFACT_ROOT/captures/<model>/<model>_session/`. The language models contain
`prefill` and `decode` stages. SmolVLA contains `prefix_encode`,
`flow_denoise`, and `action_decode`. The default inputs are synthetic; provide
documented token or image/state inputs in model2MLIR for a workload-specific
quality measurement. A successful capture verifies stage receipts, whole-bundle
integrity, and zero opaque operations; it does not prove target execution.

## 2. Audit and compile the stages

Run the target commands from the MX repository root with the environment
above. The audit is optional but gives a readable FP4/FP8 and operation-family
inventory before building large ELFs.

```sh
cd "$MX_ROOT"
for MODEL in tiny_llama deepseek_r1_qwen gemma2_2b smolvla; do
  SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
  "$CAPTURE_PY" -m tools.audit_model2mlir_session \
    --session "$SESSION" --profile "$MX_PROFILE" \
    --out "$ARTIFACT_ROOT/audits/$MODEL.json"
done

for MODEL in tiny_llama deepseek_r1_qwen gemma2_2b; do
  SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
  for STAGE in prefill decode; do
    "$CAPTURE_PY" -m tools.build_merlin_stage \
      --stage "$SESSION/stages/$STAGE" \
      --workdir "$ARTIFACT_ROOT/builds/$MODEL/$STAGE" \
      --profile "$MX_PROFILE" --merlin-src "$MERLIN_ROOT/src" \
      --llvm-install "$LLVM_INSTALL" --riscv-root "$RISCV_ROOT"
  done
done

MODEL=smolvla
SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
for STAGE in prefix_encode flow_denoise action_decode; do
  "$CAPTURE_PY" -m tools.build_merlin_stage \
    --stage "$SESSION/stages/$STAGE" \
    --workdir "$ARTIFACT_ROOT/builds/$MODEL/$STAGE" \
    --profile "$MX_PROFILE" --merlin-src "$MERLIN_ROOT/src" \
    --llvm-install "$LLVM_INSTALL" --riscv-root "$RISCV_ROOT"
done
```

Each stage writes `mx_stage_probe.json` and `model.elf` to its build
directory. The probe rejects a changed capture and missing MX catalog sites.
For the selected profile it also rejects eligible VPU operations left in the
final host MLIR. `action_decode` has no selected MX/VPU work and uses Merlin's
compiled host path. `--diagnostic-host-vpu` is a numerical control and must
not be used for a production session link.

## 3. Link each complete session

```sh
for MODEL in tiny_llama deepseek_r1_qwen gemma2_2b; do
  SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
  "$CAPTURE_PY" -m tools.link_merlin_session \
    --session-root "$SESSION" \
    --stage-build "prefill=$ARTIFACT_ROOT/builds/$MODEL/prefill" \
    --stage-build "decode=$ARTIFACT_ROOT/builds/$MODEL/decode" \
    --workdir "$ARTIFACT_ROOT/links/$MODEL" \
    --profile "$MX_PROFILE" --merlin-src "$MERLIN_ROOT/src" \
    --riscv-root "$RISCV_ROOT"
done

MODEL=smolvla
SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
"$CAPTURE_PY" -m tools.link_merlin_session \
  --session-root "$SESSION" \
  --stage-build "prefix_encode=$ARTIFACT_ROOT/builds/$MODEL/prefix_encode" \
  --stage-build "flow_denoise=$ARTIFACT_ROOT/builds/$MODEL/flow_denoise" \
  --stage-build "action_decode=$ARTIFACT_ROOT/builds/$MODEL/action_decode" \
  --workdir "$ARTIFACT_ROOT/links/$MODEL" \
  --profile "$MX_PROFILE" --merlin-src "$MERLIN_ROOT/src" \
  --riscv-root "$RISCV_ROOT"
```

Each link produces `session.elf` and `mx_session_link.json`. The linker checks
the stage roster, source and profile hashes, object hashes, shared checkpoint
weights, and persistent-state bindings. The receipt status
`linked_experimental_numeric_qualification_pending` means compilation and
linking succeeded; it is not a numerical acceptance result.

## 4. Run on Spike and inspect the receipt

This step needs a `libgemmini.so` built from the same RTL/Spike configuration
as the profile. Full checkpoint sessions can take hours on Spike.

```sh
export SPIKE_EXTLIB=/path/to/libgemmini.so
for MODEL in tiny_llama deepseek_r1_qwen gemma2_2b smolvla; do
  SESSION="$ARTIFACT_ROOT/captures/$MODEL/${MODEL}_session"
  "$CAPTURE_PY" -m tools.run_linked_merlin_session \
    --link-receipt "$ARTIFACT_ROOT/links/$MODEL/mx_session_link.json" \
    --session-root "$SESSION" --merlin-src "$MERLIN_ROOT/src" \
    --riscv-root "$RISCV_ROOT" --spike-extlib "$SPIKE_EXTLIB" \
    --out-dir "$ARTIFACT_ROOT/runs/$MODEL" \
    --timeout-s 172800 --progress-interval-s 60 \
    --perf-config e4m3_vpu --perf-set mem.host_tracking=0
done
```

While Spike runs, inspect `runs/<model>/progress.json` or watch the runner's
periodic status lines. They report completed steps, total steps, the active
program/step, and any completed step's Spike cycle and retired-instruction
counts. `spike.log` contains the guest `SESSION_BEGIN`, `SESSION_PERF`, and
`SESSION_SHA` records as they arrive.

Inspect `runs/<model>/mx_session_run.json` after the run. A structurally complete run has
`status: complete_numeric_qualification_pending` and validates every
program/step/output digest, zero memref rank mismatches, and guest `DONE`.
`--perf-config e4m3_vpu` selects Nicolas's extension's functional plus timing
mode and its VPU timing preset. `mem.host_tracking=0` skips CPU-store cache
tracking to make large model runs practical; MX/VPU commands remain timed.
Omit that override for the preset's fuller memory model after establishing a
baseline. The runner pins the mode, preset, and overrides in the receipt and
writes `gemmini_perf.txt` when the guest exits. Other `--perf-set` values can
support controlled sensitivity studies. Retain the same capture and executable
when comparing settings.

`step_performance` records each compiled model step; `program_performance`
groups them by stage; `total_performance` sums them across the whole session.
The measured interval includes arena reset,
state handoff, and model execution. It excludes ELF loading, result hashing,
and console output. Compare compiler changes with the same capture, precision
policy, timing preset and overrides, Spike binary, and extension. These are
Nicolas's modeled cycles; the fast setting excludes CPU-store cache effects,
and hardware latency claims still need RTL or FPGA calibration.
`incomplete` or a nonzero command result is a failed execution; process exit
alone is insufficient. Full-model numerical correctness still requires an
independent reference that implements the selected FP4/FP8 conversion and
MX/VPU BF16 arithmetic and checks logits or actions plus recurrent state.
The [current evidence](full_model_compilation.md) separates compiled links,
individual Spike executions, and numerical qualification.
The [paired MX+VPU attention timing check](evidence/attn_vpu_timing_20261012/README.md)
provides a smaller, reproducible performance comparison against Nicolas's
handwritten program.
The [completed SmolVLA session](evidence/smolvla_vpu_full_session_timing_20261013/README.md)
provides a full-session stage breakdown with the faster timing override.
