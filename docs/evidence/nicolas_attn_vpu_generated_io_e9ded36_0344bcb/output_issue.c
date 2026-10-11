#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_output(const void *O_hw) {
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(0))), "r"(UINT64_C(0x0010001000000400)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(256))), "r"(UINT64_C(0x0010001000000410)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(512))), "r"(UINT64_C(0x0010001000000420)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(768))), "r"(UINT64_C(0x0010001000000430)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000440)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(1280))), "r"(UINT64_C(0x0010001000000450)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(1536))), "r"(UINT64_C(0x0010001000000460)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(1792))), "r"(UINT64_C(0x0010001000000470)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000480)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(2304))), "r"(UINT64_C(0x0010001000000490)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(2560))), "r"(UINT64_C(0x00100010000004a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(2816))), "r"(UINT64_C(0x00100010000004b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(3072))), "r"(UINT64_C(0x00100010000004c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(3328))), "r"(UINT64_C(0x00100010000004d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(3584))), "r"(UINT64_C(0x00100010000004e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)O_hw + UINT64_C(3840))), "r"(UINT64_C(0x00100010000004f0)) : "memory");
}
