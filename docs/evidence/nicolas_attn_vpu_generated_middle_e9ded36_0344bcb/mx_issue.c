#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *p_scales) {
  if (((uint64_t)(uintptr_t)p_scales + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0004001800000180)), "r"(UINT64_C(0x000000003e000024)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0004030000000180)), "r"(UINT64_C(0x0000000000000108)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x000400180c000180)), "r"(UINT64_C(0x0000000000000111)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0004001800000180)), "r"(UINT64_C(0x0000000000000025)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0004030200000180)), "r"(UINT64_C(0x0000000000000109)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0000830200003020)), "r"(UINT64_C(0x0000000000000026)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x000400180c080180)), "r"(UINT64_C(0x0000000000000112)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 34, x0, %0, %1" : : "r"((((((uint64_t)(uintptr_t)p_scales + UINT64_C(0)) & UINT64_C(0x1ffffffff)) << 30) | UINT64_C(0x30a00180))), "r"(UINT64_C(0x0000000000400020)) : "memory");
}
