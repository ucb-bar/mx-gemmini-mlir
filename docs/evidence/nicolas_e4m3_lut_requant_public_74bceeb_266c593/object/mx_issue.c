#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_lut, const void *output_quantized, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales) {
  if (((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000010024)), "r"(UINT64_C(0x0001000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight_lut + UINT64_C(0))), "r"(UINT64_C(0x0000002000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation_lut + UINT64_C(0))), "r"(UINT64_C(0x0000002100000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_lut + UINT64_C(0))), "r"(UINT64_C(0x0000002200000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation_scales + UINT64_C(0))), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight_scales + UINT64_C(0))), "r"(UINT64_C(0x0000000100000080)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(16))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(32))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(48))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(1040))), "r"(UINT64_C(0x0010001000000050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(1056))), "r"(UINT64_C(0x0010001000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(1072))), "r"(UINT64_C(0x0010001000000070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(0))), "r"(UINT64_C(0x0010001000001f80)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(16))), "r"(UINT64_C(0x0010001000001f90)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(512))), "r"(UINT64_C(0x0010001000001fa0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(528))), "r"(UINT64_C(0x0010001000001fb0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1024))), "r"(UINT64_C(0x0010001000001fc0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1040))), "r"(UINT64_C(0x0010001000001fd0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1536))), "r"(UINT64_C(0x0010001000001fe0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1552))), "r"(UINT64_C(0x0010001000001ff0)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 26, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & UINT64_C(0x1ffffffff)) | UINT64_C(0x20080400000000))), "r"(UINT64_C(0x0000000000000001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000400020002)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 24, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000002000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000238)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(256))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(512))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(768))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(1280))), "r"(UINT64_C(0x0010001000000050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(1536))), "r"(UINT64_C(0x0010001000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_quantized + UINT64_C(1792))), "r"(UINT64_C(0x0010001000000070)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
}
