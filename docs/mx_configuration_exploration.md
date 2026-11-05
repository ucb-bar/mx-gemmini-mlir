# MX configuration exploration in this compiler

This repository supplies the MX dialect, target profiles, legality checks,
matrix/VPU lowering, and memory plans. The intended user entry point is
`npu-exploration`: its recipe chooses the hardware configuration, the user's
quantization provider and format for each legal site, and VPU placement. The
compiler verifies those choices against a source-bound Gemmini profile. It
does not choose a quantizer or impose a model-quality threshold.

## Inspect a selected hardware source

Use the Gemmini checkout and RV64 toolchain that match the configuration you
want to study. A v3 profile records legal operand modes, mesh and memory
geometry, coupled VPU lanes, and exact Scala source hashes:

```sh
export GEMMINI_ROOT=/path/to/gemmini
export PROFILE_OUTPUT=/path/to/new-profile-directory
python3 -m tools.export_profiles --schema v3 \
  --gemmini "$GEMMINI_ROOT" --out "$PROFILE_OUTPUT"
```

The [profile guide](current_mx_profiles.md) describes the schema and its
source checks. The [target and precision guide](configure_target_and_precision.md)
shows how to bind a model site to one of the profile's legal tuples. The
selected configurations couple VPU lanes to scratchpad row width; changing
lanes at a fixed mesh may require a different RTL configuration.

## What the current experimental path verifies

The source-bound DIM4/8/16/32 FP8 E4M3 captured matrix/VPU slice, the DIM16
FP4 E2M1 and FP6 E3M2 captured sites, and one real captured
matrix/view/sigmoid/multiply branch have functional and timed selected-Spike
full-output oracle checks. The FP6 branch also checks its live LUT words and
cache lifetime. The [physical memory plan](physical_memory_plan.md) explains
the bounded row proofs; [FP6 LUT planning](fp6_lut_memory_plan.md) records
the narrower cache proof. Spike cycles are model predictions, not calibrated
RTL or FireSim measurements.

The [native gpt-oss tile](gptoss_native_expert_tile.md) preserves the
checkpoint's packed MXFP4 weight bytes. The
[expert GEMM source stage](gptoss_native_expert_gemm.md) accepts a user-owned
activation callback. These are tile/source-stage capabilities; complete
gpt-oss routing, attention sinks, and whole-model inference are not yet
lowered through this path.

## Integration boundary

Merlin supplies the captured graph and host path; this repository checks and
lowers selected MX sites. The [NPU VPU rewrite](npu_vpu_stage_bridge.md)
records which source regions were selected and which stayed on host. The
current captured slice can still use an explicit host bridge between matrix
and VPU sites. A complete model run requires all operations, memory effects,
and user-selected formats to pass the compiler stages and selected Spike
oracle. NPU must pin compatible Merlin and MX commits as submodules before a
fresh recursive checkout reproduces this experimental path.

The selected development source tuple used for the current Spike evidence is
Gemmini `12912323`, RoCC tests `73abfde`, this compiler at `95c63645`, and
Merlin `739c457ed`. These are exact-source identities for the evidence, not
claims that the source tuple is already available from every upstream branch.
