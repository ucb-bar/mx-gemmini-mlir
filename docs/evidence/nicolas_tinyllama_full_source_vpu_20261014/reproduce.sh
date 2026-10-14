#!/usr/bin/env bash
set -euo pipefail

# Run after the neighboring full-source reproduce.sh has generated and run the
# scalar-host baseline in the same task-local WORK_DIR.
: "${WORK_DIR:?completed full-source reproduction directory}"
: "${ROCC_ROOT:?pinned Gemmini rocc-tests directory}"
: "${RISCV_ROOT:?RISC-V toolchain and Spike root}"
: "${SPIKE_EXTLIB:?pinned MX+VPU Spike extension}"
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$WORK_DIR/source"
OUT="$WORK_DIR/out_vpu/baremetal"
test -f "$SRC/baremetal/mxgemmini/src/llama_model.c"
test -f "$SRC/baremetal/mxgemmini/data/llama_model_m64.bin"
test -f "$WORK_DIR/out/baremetal/spike/llama_model_m64"
patch -s --forward -p1 -d "$SRC" < "$HERE/vpu_residual.patch"

CPATH="$RISCV_ROOT/include/riscv" make -j1 -B -C \
  "$SRC/baremetal/mxgemmini" ROCC="$ROCC_ROOT" \
  CC="$RISCV_ROOT/bin/riscv64-unknown-elf-gcc" \
  OBJCOPY="$RISCV_ROOT/bin/riscv64-unknown-elf-objcopy" \
  OUT="$OUT" \
  EXTRA_llama_model_m64='-DLLAMA_VPU_RESIDUAL -DLLAMA_STAGE_TIMING -DLLAMA_DUMP_LAST_LOGITS' \
  "$OUT/spike/llama_model_m64" > "$WORK_DIR/build_vpu.log" 2>&1

"$RISCV_ROOT/bin/spike" --isa=rv64gc_zicntr \
  --extlib="$SPIKE_EXTLIB" --extension=gemmini \
  -m0x80000000:0x2c0000000 "$OUT/spike/llama_model_m64" \
  > "$WORK_DIR/spike_vpu.log" 2>&1
rg '^(stage_cycles64|llama MODEL benchmark)' "$WORK_DIR/spike_vpu.log"
