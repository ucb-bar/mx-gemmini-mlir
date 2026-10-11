#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *Q_SCALES_FOLDED, const void *KT_SCALES, const void *Q_IN, const void *KT_IN, const void *scale_sink, const void *S0_HW) {
  if (((uint64_t)(uintptr_t)Q_SCALES_FOLDED + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)KT_SCALES + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)scale_sink + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)Q_SCALES_FOLDED + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x400000000000))), "r"(UINT64_C(0x0000a00000000040)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_IN + UINT64_C(0))), "r"(UINT64_C(0x0010004000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_IN + UINT64_C(1024))), "r"(UINT64_C(0x0010004000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_IN + UINT64_C(2048))), "r"(UINT64_C(0x0010004000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_IN + UINT64_C(3072))), "r"(UINT64_C(0x00100040000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_IN + UINT64_C(0))), "r"(UINT64_C(0x0010002000001000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_IN + UINT64_C(2048))), "r"(UINT64_C(0x0010002000001020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_IN + UINT64_C(4096))), "r"(UINT64_C(0x0010002000001040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_IN + UINT64_C(6144))), "r"(UINT64_C(0x0010002000001060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x000000000001c004)), "r"(UINT64_C(0x0001000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)KT_SCALES + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0040800100000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 26, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)scale_sink + UINT64_C(0)) & UINT64_C(0x1ffffffff)) | UINT64_C(0x1020080800000000))), "r"(UINT64_C(0x0000000000060001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000400020004)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 24, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000001080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000200000000338)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(0))), "r"(UINT64_C(0x0010001000002000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(256))), "r"(UINT64_C(0x0010001000002010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(512))), "r"(UINT64_C(0x0010001000002020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(768))), "r"(UINT64_C(0x0010001000002030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(1024))), "r"(UINT64_C(0x0010001000002040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(1280))), "r"(UINT64_C(0x0010001000002050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(1536))), "r"(UINT64_C(0x0010001000002060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(1792))), "r"(UINT64_C(0x0010001000002070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(2048))), "r"(UINT64_C(0x0010001000002080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(2304))), "r"(UINT64_C(0x0010001000002090)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(2560))), "r"(UINT64_C(0x00100010000020a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(2816))), "r"(UINT64_C(0x00100010000020b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(3072))), "r"(UINT64_C(0x00100010000020c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(3328))), "r"(UINT64_C(0x00100010000020d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(3584))), "r"(UINT64_C(0x00100010000020e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)S0_HW + UINT64_C(3840))), "r"(UINT64_C(0x00100010000020f0)) : "memory");
}
