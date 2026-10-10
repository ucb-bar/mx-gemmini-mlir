#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *score, const void *output) {
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)score + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)score + UINT64_C(256))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)score + UINT64_C(512))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)score + UINT64_C(768))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001010000000000)), "r"(UINT64_C(0x0000000000000088)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001020004000000)), "r"(UINT64_C(0x0000000000000091)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001020000002000)), "r"(UINT64_C(0x0000000000000025)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0001030000002000)), "r"(UINT64_C(0x0000000000000089)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x0000430000003000)), "r"(UINT64_C(0x0000000000000026)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 33, x0, %0, %1" : : "r"(UINT64_C(0x000100400c002000)), "r"(UINT64_C(0x0000000000000092)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output + UINT64_C(0))), "r"(UINT64_C(0x0010001000000400)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output + UINT64_C(256))), "r"(UINT64_C(0x0010001000000410)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output + UINT64_C(512))), "r"(UINT64_C(0x0010001000000420)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output + UINT64_C(768))), "r"(UINT64_C(0x0010001000000430)) : "memory");
}
