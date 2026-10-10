#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(void) {
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001010000000000)), "r"(UINT64_C(0x0000000000000088)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001020004000000)), "r"(UINT64_C(0x0000000000000091)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001020000002000)), "r"(UINT64_C(0x0000000000000025)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001030000002000)), "r"(UINT64_C(0x0000000000000089)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0000430000003000)), "r"(UINT64_C(0x0000000000000026)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x000100400c002000)), "r"(UINT64_C(0x0000000000000092)) : "memory");
}
