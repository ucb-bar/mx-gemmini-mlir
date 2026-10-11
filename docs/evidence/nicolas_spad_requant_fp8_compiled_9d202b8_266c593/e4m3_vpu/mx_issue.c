#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *X, const void *scales_hw, const void *scales_hw2, const void *codes_flat_hw, const void *codes_tiled_hw) {
  if (((uint64_t)(uintptr_t)scales_hw + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)scales_hw2 + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(256))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(512))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(768))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(1280))), "r"(UINT64_C(0x0010001000000050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(1536))), "r"(UINT64_C(0x0010001000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(1792))), "r"(UINT64_C(0x0010001000000070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(2304))), "r"(UINT64_C(0x0010001000000090)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(2560))), "r"(UINT64_C(0x00100010000000a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(2816))), "r"(UINT64_C(0x00100010000000b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(3072))), "r"(UINT64_C(0x00100010000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(3328))), "r"(UINT64_C(0x00100010000000d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(3584))), "r"(UINT64_C(0x00100010000000e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)X + UINT64_C(3840))), "r"(UINT64_C(0x00100010000000f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 34, x0, %0, %1" : : "r"((((((uint64_t)(uintptr_t)scales_hw + UINT64_C(0)) & UINT64_C(0x1ffffffff)) << 30) | UINT64_C(0x4000000))), "r"(UINT64_C(0x0000000000400020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(0))), "r"(UINT64_C(0x0010001000001000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(256))), "r"(UINT64_C(0x0010001000001010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(512))), "r"(UINT64_C(0x0010001000001020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(768))), "r"(UINT64_C(0x0010001000001030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(1024))), "r"(UINT64_C(0x0010001000001040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(1280))), "r"(UINT64_C(0x0010001000001050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(1536))), "r"(UINT64_C(0x0010001000001060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_flat_hw + UINT64_C(1792))), "r"(UINT64_C(0x0010001000001070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 34, x0, %0, %1" : : "r"((((((uint64_t)(uintptr_t)scales_hw2 + UINT64_C(0)) & UINT64_C(0x1ffffffff)) << 30) | UINT64_C(0x18000000))), "r"(UINT64_C(0x0000000000400020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(0))), "r"(UINT64_C(0x0010001000002000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(256))), "r"(UINT64_C(0x0010001000002010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(512))), "r"(UINT64_C(0x0010001000002020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(768))), "r"(UINT64_C(0x0010001000002030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(1024))), "r"(UINT64_C(0x0010001000002040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(1280))), "r"(UINT64_C(0x0010001000002050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(1536))), "r"(UINT64_C(0x0010001000002060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)codes_tiled_hw + UINT64_C(1792))), "r"(UINT64_C(0x0010001000002070)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
}
