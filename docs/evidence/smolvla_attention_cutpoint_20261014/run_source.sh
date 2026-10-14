#!/usr/bin/env bash
set -euo pipefail

: "${NPU_ROOT:?Set NPU_ROOT to pinned npu-exploration 0344bcb}"
: "${ROCC_ROOT:?Set ROCC_ROOT to the matching gemmini-rocc-tests checkout}"
: "${RISCV_CC:?Set RISCV_CC to riscv64-unknown-elf-gcc}"
: "${SPIKE:?Set SPIKE to Nicolas's Spike executable}"
: "${SPIKE_EXTENSION:?Set SPIKE_EXTENSION to Nicolas's libgemmini.so}"
: "${OUT_DIR:?Set OUT_DIR to a scratch directory for this run}"

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$ROCC_ROOT/riscv-tests/benchmarks/common"
RISCV_ENV="${RISCV_ENV:-$ROCC_ROOT/riscv-tests/env}"
mkdir -p "$OUT_DIR"
cp "$NPU_ROOT/baremetal/mxgemmini/src/attn_flash.c" "$OUT_DIR/attn_flash_smol.c"
printf '%s  %s\n' \
  004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0 \
  "$OUT_DIR/attn_flash_smol.c" | sha256sum --check --status
patch -s "$OUT_DIR/attn_flash_smol.c" "$HERE/attn_flash_smol.patch"
cp "$HERE/driver.c" "$OUT_DIR/"
cp "${SOURCE_OPERAND_HEADER:-$HERE/smol_head0.h}" "$OUT_DIR/smol_head0.h"
cp "${SOURCE_EXPECT_HEADER:-$HERE/smol_expect.h}" "$OUT_DIR/smol_expect.h"

"$RISCV_CC" -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 \
  -O2 -ffast-math -fno-common -fno-builtin-printf \
  -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc \
  -I"$OUT_DIR" -I"$ROCC_ROOT" -I"$ROCC_ROOT/riscv-tests" \
  -I"$RISCV_ENV" -I"$BENCH" -DID_STRING= -DPRINT_TILE=0 \
  -nostdlib -nostartfiles -static -T "$BENCH/test.ld" \
  -DBAREMETAL=1 -DSPIKE_SIM "$OUT_DIR/driver.c" \
  -o "$OUT_DIR/smol_head0.elf" "$BENCH/crt.S" "$BENCH/syscalls.c" \
  -lm -lgcc

GEMMINI_MODE=both "$SPIKE" --isa=rv64gc_zicntr \
  --extlib="$SPIKE_EXTENSION" --extension=gemmini \
  -m0x80000000:0x40000000 "$OUT_DIR/smol_head0.elf" \
  > "$OUT_DIR/source_spike.log" 2>&1
grep -E 'check O hash|accuracy O head|PERF |attn_flash PASSED' \
  "$OUT_DIR/source_spike.log"
