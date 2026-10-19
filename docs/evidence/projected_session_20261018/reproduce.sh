#!/usr/bin/env bash
set -euo pipefail

: "${TMPDIR:?Set TMPDIR to the task-local temporary directory}"
: "${LLVM_INSTALL:?Set LLVM_INSTALL to the selected LLVM installation}"
: "${RISCV_ROOT:?Set RISCV_ROOT to the selected RV64 toolchain}"
: "${MERLIN_COMPILER_PYTHON:?Select the compiler interpreter with torch-mlir}"
capture_python="${CAPTURE_PYTHON:-python}"
fixture_root="${1:-$TMPDIR/projected-session-reproduction}"
merlin_src="${MERLIN_SRC:-../merlin/src}"
profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
extension=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$merlin_src:../model2mlir:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MERLIN_READELF="$LLVM_INSTALL/bin/llvm-readelf"

"$capture_python" -m tools.prepare_projected_session_fixture \
  --out-dir "$fixture_root/capture" --profile "$profile"
for mode in original projected; do
  projection_flags=()
  if [[ "$mode" == projected ]]; then
    projection_flags=(--prune-unused-checkpoint-weights)
  fi
  for stage in prefill decode; do
    "$capture_python" -m tools.build_merlin_stage \
      --stage "$fixture_root/capture/stages/$stage" \
      --workdir "$fixture_root/${mode}_$stage" \
      --profile "$profile" --merlin-src "$merlin_src" \
      --llvm-install "$LLVM_INSTALL" --riscv-root "$RISCV_ROOT" \
      --execution-policy "$fixture_root/capture/${stage}_policy.json" \
      --prepack-checkpoint-weights --structural-elf-admission \
      --spike-arena-base 0xd0000000 "${projection_flags[@]}"
  done
  "$capture_python" -m tools.link_merlin_session \
    --session-root "$fixture_root/capture" \
    --stage-build "prefill=$fixture_root/${mode}_prefill" \
    --stage-build "decode=$fixture_root/${mode}_decode" \
    --workdir "$fixture_root/${mode}_linked" \
    --profile "$profile" --merlin-src "$merlin_src" --riscv-root "$RISCV_ROOT" \
    --structural-elf-admission --measure-preparation \
    --dump-output prefill:0 --dump-output prefill:1 \
    --dump-output decode:0 --dump-output decode:1
  "$capture_python" -m tools.run_linked_merlin_session \
    --link-receipt "$fixture_root/${mode}_linked/mx_session_link.json" \
    --session-root "$fixture_root/capture" --merlin-src "$merlin_src" \
    --riscv-root "$RISCV_ROOT" --spike-extlib "$extension" \
    --out-dir "$fixture_root/${mode}_run" --timeout-s 600 \
    --perf-config e4m3_vpu --perf-set mem.host_tracking=0
done
