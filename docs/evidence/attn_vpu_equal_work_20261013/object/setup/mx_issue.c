#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_setup(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES, const void *Q_IN, const void *KT_IN, const void *V_IN, const void *scale_sink, const void *S_hw, const void *p_scales, const void *O_hw, const void *P_hw, const void *p_codes_hw) {
  if (((uint64_t)(uintptr_t)V_SCALES + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)Q_SCALES + UINT64_C(0))), "r"(UINT64_C(0x0000000000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)KT_SCALES + UINT64_C(0))), "r"(UINT64_C(0x0000000100000400)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)V_SCALES + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0002200100000080)) : "memory");
}
