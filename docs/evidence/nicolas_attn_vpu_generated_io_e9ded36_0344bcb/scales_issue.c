#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_scales(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES) {
  if (((uint64_t)(uintptr_t)V_SCALES + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_SCALES + UINT64_C(0))), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_SCALES + UINT64_C(0))), "r"(UINT64_C(0x0000000100000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)V_SCALES + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x400000000000))), "r"(UINT64_C(0x0000a00100000040)) : "memory");
}
