#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_probability_codes(const void *p_codes_hw) {
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(0))), "r"(UINT64_C(0x0010001000000280)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(256))), "r"(UINT64_C(0x0010001000000290)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(512))), "r"(UINT64_C(0x00100010000002a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(768))), "r"(UINT64_C(0x00100010000002b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(1024))), "r"(UINT64_C(0x00100010000002c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(1280))), "r"(UINT64_C(0x00100010000002d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(1536))), "r"(UINT64_C(0x00100010000002e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)p_codes_hw + UINT64_C(1792))), "r"(UINT64_C(0x00100010000002f0)) : "memory");
}
