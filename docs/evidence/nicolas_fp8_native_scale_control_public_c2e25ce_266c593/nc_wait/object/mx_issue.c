#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue(const void *activation, const void *activation_scales, const void *output_bf16, const void *scratch_output_scales, const void *weight, const void *weight_scales) {
  if (((uint64_t)(uintptr_t)activation_scales + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)weight_scales + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)weight_scales + UINT64_C(64)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x000000000001c004)), "r"(UINT64_C(0x0001000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)activation_scales + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)weight_scales + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001000100000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)activation_scales + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001040000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)weight_scales + UINT64_C(64)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001020100000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100109)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 26, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & UINT64_C(0x1ffffffff)) | UINT64_C(0x80101000000000))), "r"(UINT64_C(0x0000000000010001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000800040008)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 10, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(0))), "r"(((uint64_t)(uintptr_t)weight + UINT64_C(0))) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 11, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(0))) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 12, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000080)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 13, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000050000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000800040008)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 10, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(((uint64_t)(uintptr_t)weight + UINT64_C(64))) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 11, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(128))) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 12, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000080)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 13, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000060000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
}
