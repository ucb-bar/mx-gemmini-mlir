#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_bf16, const void *output_lut, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales) {
  if (((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x000000000001d824)), "r"(UINT64_C(0x0001000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001800000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001900000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001a00000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation_scales + UINT64_C(0))), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight_scales + UINT64_C(0))), "r"(UINT64_C(0x0000000100000080)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000200101)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(0))), "r"(UINT64_C(0x0020002000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(32))), "r"(UINT64_C(0x0020002000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000200101)), "r"(UINT64_C(0x0000000000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(0))), "r"(UINT64_C(0x0020002000001fc0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1024))), "r"(UINT64_C(0x0020002000001fe0)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 26, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & UINT64_C(0x1ffffffff)) | UINT64_C(0x10040200000000))), "r"(UINT64_C(0x0000000000000001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000200010001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 24, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000002000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000004000000238)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(0))), "r"(UINT64_C(0x0020002000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(1024))), "r"(UINT64_C(0x0020002000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(2048))), "r"(UINT64_C(0x0020002000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(3072))), "r"(UINT64_C(0x00200020000000a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(4096))), "r"(UINT64_C(0x00200020000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(5120))), "r"(UINT64_C(0x00200020000000e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(6144))), "r"(UINT64_C(0x0020002000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(7168))), "r"(UINT64_C(0x0020002000000120)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
}
