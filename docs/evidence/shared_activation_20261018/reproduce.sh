#!/usr/bin/env bash
set -euo pipefail
# Run from the editable mx-gemmini-mlir worktree. All products stay in the task.
: "${TMPDIR:?Set TMPDIR to the task .tmp directory}"
: "${MX_REPRO_DIR:?Set MX_REPRO_DIR to a fresh directory inside the task}"
: "${MX_SOURCE_STAGE:?Set MX_SOURCE_STAGE to the matched 160-token capture}"
: "${MERLIN_COMPILER_PYTHON:?Select the compiler Python}"
: "${MERLIN_MLIR_INSTALL:?Select the LLVM install}"
: "${MERLIN_RISCV_ROOT:?Select the existing RV64 toolchain root}"
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu GEMMINI_PERF_SET=mem.host_tracking=0
export MERLIN_CLANG="$MERLIN_MLIR_INSTALL/bin/clang"
export MERLIN_READELF="$MERLIN_MLIR_INSTALL/bin/llvm-readelf"
python=${MX_PYTHON:-$PWD/../.tmp/smolvla-venv/bin/python}
profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
extension=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
for rows in 160 1; do
  probe="$MX_REPRO_DIR/qkv$rows"
  "$python" -m tools.capture_tinyllama_projections --stage "$MX_SOURCE_STAGE" \
    --out-dir "$probe" --rows "$rows" --profile "$profile"
  for variant in baseline shared retain_rhs; do
    options=()
    if [[ $variant != baseline ]]; then options+=(--share-activation-packing); fi
    if [[ $variant == retain_rhs ]]; then options+=(--retain-shared-activation-rhs); fi
    "$python" -m tools.build_merlin_stage --stage "$probe/capture" --workdir "$probe/$variant" \
      --profile "$profile" --merlin-src ../merlin/src --llvm-install "$MERLIN_MLIR_INSTALL" \
      --riscv-root "$MERLIN_RISCV_ROOT" --execution-policy "$probe/policy.json" \
      --prepack-checkpoint-weights --prune-unused-checkpoint-weights --structural-elf-admission \
      --spike-extlib "$extension" --spike-timeout 1800 --output-dump-cap 1048576 "${options[@]}"
  done
  "$python" -m tools.compare_shared_preparation --stage "$probe/capture" \
    --build "baseline=$probe/baseline" --build "shared=$probe/shared" \
    --build "retain_rhs=$probe/retain_rhs" --out "$probe/comparison.json"
done
# Independent tail and budget cases use capture_shared_projection_probe --rows 19.
# Build baseline/shared as above; add --activation-packing-budget-bytes 1024 to
# a third shared build and compare it under the label budget_refused.
