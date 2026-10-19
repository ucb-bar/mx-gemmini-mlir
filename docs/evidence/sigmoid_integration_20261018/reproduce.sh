#!/usr/bin/env bash
set -eu
: "${PYTHON:?set PYTHON to the capture/compiler Python}"
: "${MERLIN_SRC:?set MERLIN_SRC to the Merlin source directory}"
: "${MERLIN_MLIR_INSTALL:?set MERLIN_MLIR_INSTALL}"
: "${RISCV_ROOT:?set RISCV_ROOT}"
result_dir=${1:?provide a fresh output directory}
profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
extension=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
export GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu GEMMINI_PERF_SET=mem.host_tracking=0
"$PYTHON" -m tools.capture_sigmoid_projection_probe --out-dir "$result_dir" --profile "$profile"
common=(--stage "$result_dir/capture" --profile "$profile" --merlin-src "$MERLIN_SRC"
  --llvm-install "$MERLIN_MLIR_INSTALL" --riscv-root "$RISCV_ROOT"
  --prepack-checkpoint-weights --prune-unused-checkpoint-weights --connect-vpu-multiplies
  --execution-policy "$result_dir/policy.json" --structural-elf-admission
  --spike-extlib "$extension" --output-dump-cap 703 --spike-timeout 900)
"$PYTHON" -m tools.build_merlin_stage "${common[@]}" --workdir "$result_dir/baseline"
"$PYTHON" -m tools.build_merlin_stage "${common[@]}" --workdir "$result_dir/connected" --connect-vpu-sigmoid-multiplies
"$PYTHON" -m tools.compare_shared_preparation --stage "$result_dir/capture" \
  --build "baseline=$result_dir/baseline" --build "connected=$result_dir/connected" --out "$result_dir/comparison.json"
