#!/usr/bin/env bash
set -euo pipefail

# Run this against the pinned repositories named in README.md. All outputs
# stay in WORK_DIR; INPUT_IDS is the captured model2MLIR prefill inputs.npz.
: "${NPU_ROOT:?}"
: "${ROCC_ROOT:?}"
: "${RISCV_ROOT:?}"
: "${RISCV_TEST_ENV:?}"
: "${SPIKE_EXTLIB:?}"
: "${INPUT_IDS:?}"
: "${WORK_DIR:?}"
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NPU_SRC_ROOT="$NPU_ROOT/baremetal/mxgemmini"
BENCH_ROOT="$ROCC_ROOT/riscv-tests/benchmarks/common"
mkdir -p "$WORK_DIR"

cp "$NPU_SRC_ROOT/gen/gen_llama_layer_e2e.py" "$WORK_DIR/gen_llama_layer_e2e.py"
patch -s -p1 -d "$WORK_DIR" < "$SCRIPT_DIR/capture_input.patch"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH="$NPU_SRC_ROOT/gen" "$PYTHON_BIN" -u \
  "$WORK_DIR/gen_llama_layer_e2e.py" --sq 64 --sk 64 --layer 5 \
  --input-ids "$INPUT_IDS"

(
  cd "$WORK_DIR"
  "$RISCV_ROOT/bin/riscv64-unknown-elf-objcopy" -I binary \
    -O elf64-littleriscv -B riscv --set-section-alignment .data=64 \
    llama_layer_e2e.bin llama_layer_e2e_blob.o
)
"$RISCV_ROOT/bin/riscv64-unknown-elf-gcc" \
  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gc -Wa,-march=rv64gc -nostdlib -nostartfiles -static \
  -T "$BENCH_ROOT/test.ld" -DBAREMETAL=1 -DSPIKE_SIM -DBK_FIRST=64 \
  -DID_STRING= -DPRINT_TILE=0 \
  -I"$WORK_DIR" -I"$NPU_SRC_ROOT/include" -I"$ROCC_ROOT" \
  -I"$ROCC_ROOT/riscv-tests" -I"$RISCV_TEST_ENV" -I"$BENCH_ROOT" \
  "$NPU_SRC_ROOT/src/llama_layer_e2e.c" \
  "$BENCH_ROOT/syscalls.c" "$BENCH_ROOT/crt.S" \
  "$WORK_DIR/llama_layer_e2e_blob.o" -o "$WORK_DIR/llama_layer_e2e.elf" \
  -lm -lgcc
"$RISCV_ROOT/bin/spike" --isa=rv64gc_zicntr \
  --extlib="$SPIKE_EXTLIB" --extension=gemmini \
  -m0x80000000:0x2c0000000 "$WORK_DIR/llama_layer_e2e.elf" \
  > "$WORK_DIR/spike.log" 2>&1
rg '^(accuracy  h_out|PERF layer:|llama_layer_e2e )' "$WORK_DIR/spike.log"
