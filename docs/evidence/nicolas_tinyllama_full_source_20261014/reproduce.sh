#!/usr/bin/env bash
set -euo pipefail

# Reproduce the 22-layer plus head source run in WORK_DIR. No generated data is
# written into the source checkouts.
: "${NPU_ROOT:?pinned npu-exploration checkout}"
: "${ROCC_ROOT:?gemmini-rocc-tests directory in pinned Gemmini checkout}"
: "${MXQ_ROOT:?pinned microscaling-quant checkout}"
: "${RISCV_ROOT:?RISC-V toolchain and Spike root}"
: "${SPIKE_EXTLIB:?pinned MX+VPU libgemmini.so}"
: "${INPUT_IDS:?model2MLIR TinyLlama prefill inputs.npz}"
: "${WORK_DIR:?task-local scratch directory}"
PYTHON_BIN=${PYTHON_BIN:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$WORK_DIR/source"
OUT="$WORK_DIR/out/baremetal"
NPU_COMMIT=0344bcba70b899039ccd0ac0801ca5605d46d3eb
GEMMINI_COMMIT=266c593f2cb51d7e3fe83fc0317072b585ac3c52

test "$(git -C "$NPU_ROOT" rev-parse HEAD)" = "$NPU_COMMIT"
test "$(git -C "$ROCC_ROOT" rev-parse HEAD)" = "$GEMMINI_COMMIT"
if test -e "$SRC/kernels/captures/llama_model.py"; then
  echo "WORK_DIR already contains a source copy; choose a fresh directory" >&2
  exit 1
fi
mkdir -p "$SRC" "$WORK_DIR/software/gemmini-rocc-tests"
git -C "$NPU_ROOT" archive "$NPU_COMMIT" \
  kernels/captures kernels/host_math.py baremetal/mxgemmini/gen \
  baremetal/mxgemmini/src baremetal/mxgemmini/include \
  baremetal/mxgemmini/Makefile compiler models config | tar -x -C "$SRC"
mkdir -p "$SRC/microscaling-quant"
cp -a "$MXQ_ROOT/mxq" "$SRC/microscaling-quant/mxq"
cp "$ROCC_ROOT/gen_matmul_llama.py" \
  "$WORK_DIR/software/gemmini-rocc-tests/gen_matmul_llama.py"
patch -s -p1 -d "$SRC" < "$HERE/../nicolas_tinyllama_two_layer_source_20261014/captured_ids.patch"
patch -s -p1 -d "$SRC" < "$HERE/../nicolas_tinyllama_two_layer_source_20261014/limit_workers.patch"
patch -s -p1 -d "$SRC" < "$HERE/diagnostic_prints.patch"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$SRC" \
  "$PYTHON_BIN" -u -m kernels.captures.llama_model \
  --seq 64 --input-ids "$INPUT_IDS" \
  --out "$WORK_DIR/capture22" > "$WORK_DIR/capture.log" 2>&1

(
  cd "$SRC/baremetal/mxgemmini/gen"
  NPU_EXPLORATION="$SRC" PYTHONPATH="$SRC" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 "$PYTHON_BIN" -u gen_llama_model.py \
    --layers 22 --capture "$WORK_DIR/capture22" \
    --tag _m64 > "$WORK_DIR/generate.log" 2>&1
)

CPATH="$RISCV_ROOT/include/riscv" make -j1 -C \
  "$SRC/baremetal/mxgemmini" ROCC="$ROCC_ROOT" \
  CC="$RISCV_ROOT/bin/riscv64-unknown-elf-gcc" \
  OBJCOPY="$RISCV_ROOT/bin/riscv64-unknown-elf-objcopy" \
  OUT="$OUT" EXTRA_llama_model_m64=-DLLAMA_DUMP_LAST_LOGITS \
  "$OUT/spike/llama_model_m64" \
  > "$WORK_DIR/build.log" 2>&1
"$RISCV_ROOT/bin/spike" --isa=rv64gc_zicntr \
  --extlib="$SPIKE_EXTLIB" --extension=gemmini \
  -m0x80000000:0x2c0000000 "$OUT/spike/llama_model_m64" \
  > "$WORK_DIR/spike.log" 2>&1
rg '^(layer  |grade logits|cycles64|llama MODEL test)' "$WORK_DIR/spike.log"
