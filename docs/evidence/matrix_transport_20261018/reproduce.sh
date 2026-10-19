#!/usr/bin/env bash
set -euo pipefail
# Run from the editable mx-gemmini-mlir task worktree.
: "${TMPDIR:?Set TMPDIR to the task .tmp directory}"
: "${MX_REPRO_DIR:?Set MX_REPRO_DIR to a fresh directory inside the task}"
: "${MX_SOURCE_STAGE:?Set MX_SOURCE_STAGE to the matched 160-token capture}"
: "${MERLIN_COMPILER_PYTHON:?Select the compiler Python}"
: "${MERLIN_MLIR_INSTALL:?Select the LLVM install}"
: "${MERLIN_RISCV_ROOT:?Select the existing RV64 toolchain root}"
[[ ! -e $MX_REPRO_DIR ]]
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu GEMMINI_PERF_SET=mem.host_tracking=0
export MERLIN_CLANG="$MERLIN_MLIR_INSTALL/bin/clang"
export MERLIN_READELF="$MERLIN_MLIR_INSTALL/bin/llvm-readelf"
python=${MX_PYTHON:-$PWD/../.tmp/smolvla-venv/bin/python}
profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
extension=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
for name in tail19 bias1 mlp160 mlp1 shared; do
  probe="$MX_REPRO_DIR/$name"
  capture_options=()
  build_options=()
  case $name in
    mlp160|mlp1)
      rows=${name#mlp}
      "$python" -m tools.capture_tinyllama_residual --stage "$MX_SOURCE_STAGE" \
        --out-dir "$probe" --rows "$rows" --profile "$profile"
      ;;
    *)
      if [[ $name == bias1 ]]; then capture_options+=(--rows 1 --bias-broadcast); fi
      if [[ $name == shared ]]; then
        capture_options+=(--shared-projections)
        build_options+=(--share-activation-packing)
      fi
      "$python" -m tools.capture_matrix_residual_probe --out-dir "$probe" \
        --profile "$profile" "${capture_options[@]}"
      ;;
  esac
  for variant in baseline transport; do
    transport_options=()
    if [[ $variant == transport ]]; then transport_options+=(--matrix-vpu-transport); fi
    "$python" -m tools.build_merlin_stage --stage "$probe/capture" --workdir "$probe/$variant" \
      --profile "$profile" --merlin-src ../merlin/src --llvm-install "$MERLIN_MLIR_INSTALL" \
      --riscv-root "$MERLIN_RISCV_ROOT" --execution-policy "$probe/policy.json" \
      --prepack-checkpoint-weights --prune-unused-checkpoint-weights --structural-elf-admission \
      --spike-extlib "$extension" --spike-timeout 1800 --output-dump-cap 1048576 \
      "${build_options[@]}" "${transport_options[@]}"
  done
  "$python" -m tools.compare_shared_preparation --stage "$probe/capture" \
    --build "baseline=$probe/baseline" --build "transport=$probe/transport" \
    --out "$probe/comparison.json"
done
