#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

static void mx_issue_commands(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_bf16, const void *output_lut, const void *output_quantized, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales) {
  if (((uint64_t)(uintptr_t)activation_scales + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)weight_scales + UINT64_C(0)) & ~UINT64_C(0xffffffffff)) __builtin_trap();
  if (((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & ~UINT64_C(0x1ffffffff)) __builtin_trap();
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x000000000001d424)), "r"(UINT64_C(0x0001000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100109)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000100)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001800000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001900000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 29, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_lut + UINT64_C(0))), "r"(UINT64_C(0x0000001a00000040)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)activation_scales + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)weight_scales + UINT64_C(0)) & UINT64_C(0xffffffffff)) | UINT64_C(0x800000000000))), "r"(UINT64_C(0x0001000100000080)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(16))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(32))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(48))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(64))), "r"(UINT64_C(0x0010001000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(80))), "r"(UINT64_C(0x0010001000000050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(96))), "r"(UINT64_C(0x0010001000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(112))), "r"(UINT64_C(0x0010001000000070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2064))), "r"(UINT64_C(0x0010001000000090)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2080))), "r"(UINT64_C(0x00100010000000a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2096))), "r"(UINT64_C(0x00100010000000b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2112))), "r"(UINT64_C(0x00100010000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2128))), "r"(UINT64_C(0x00100010000000d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2144))), "r"(UINT64_C(0x00100010000000e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(2160))), "r"(UINT64_C(0x00100010000000f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4096))), "r"(UINT64_C(0x0010001000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4112))), "r"(UINT64_C(0x0010001000000110)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4128))), "r"(UINT64_C(0x0010001000000120)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4144))), "r"(UINT64_C(0x0010001000000130)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4160))), "r"(UINT64_C(0x0010001000000140)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4176))), "r"(UINT64_C(0x0010001000000150)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4192))), "r"(UINT64_C(0x0010001000000160)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(4208))), "r"(UINT64_C(0x0010001000000170)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6144))), "r"(UINT64_C(0x0010001000000180)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6160))), "r"(UINT64_C(0x0010001000000190)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6176))), "r"(UINT64_C(0x00100010000001a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6192))), "r"(UINT64_C(0x00100010000001b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6208))), "r"(UINT64_C(0x00100010000001c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6224))), "r"(UINT64_C(0x00100010000001d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6240))), "r"(UINT64_C(0x00100010000001e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)activation + UINT64_C(6256))), "r"(UINT64_C(0x00100010000001f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(0))), "r"(UINT64_C(0x0010001000003e00)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(16))), "r"(UINT64_C(0x0010001000003e10)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(32))), "r"(UINT64_C(0x0010001000003e20)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(48))), "r"(UINT64_C(0x0010001000003e30)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1024))), "r"(UINT64_C(0x0010001000003e40)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1040))), "r"(UINT64_C(0x0010001000003e50)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1056))), "r"(UINT64_C(0x0010001000003e60)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(1072))), "r"(UINT64_C(0x0010001000003e70)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(2048))), "r"(UINT64_C(0x0010001000003e80)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(2064))), "r"(UINT64_C(0x0010001000003e90)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(2080))), "r"(UINT64_C(0x0010001000003ea0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(2096))), "r"(UINT64_C(0x0010001000003eb0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(3072))), "r"(UINT64_C(0x0010001000003ec0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(3088))), "r"(UINT64_C(0x0010001000003ed0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(3104))), "r"(UINT64_C(0x0010001000003ee0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(3120))), "r"(UINT64_C(0x0010001000003ef0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(4096))), "r"(UINT64_C(0x0010001000003f00)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(4112))), "r"(UINT64_C(0x0010001000003f10)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(4128))), "r"(UINT64_C(0x0010001000003f20)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(4144))), "r"(UINT64_C(0x0010001000003f30)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(5120))), "r"(UINT64_C(0x0010001000003f40)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(5136))), "r"(UINT64_C(0x0010001000003f50)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(5152))), "r"(UINT64_C(0x0010001000003f60)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(5168))), "r"(UINT64_C(0x0010001000003f70)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(6144))), "r"(UINT64_C(0x0010001000003f80)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(6160))), "r"(UINT64_C(0x0010001000003f90)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(6176))), "r"(UINT64_C(0x0010001000003fa0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(6192))), "r"(UINT64_C(0x0010001000003fb0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(7168))), "r"(UINT64_C(0x0010001000003fc0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(7184))), "r"(UINT64_C(0x0010001000003fd0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(7200))), "r"(UINT64_C(0x0010001000003fe0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)weight + UINT64_C(7216))), "r"(UINT64_C(0x0010001000003ff0)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 26, x0, %0, %1" : : "r"(((((uint64_t)(uintptr_t)scratch_output_scales + UINT64_C(0)) & UINT64_C(0x1ffffffff)) | UINT64_C(0x40100800000000))), "r"(UINT64_C(0x0000000000000001)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 9, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000800040004)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 24, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000004000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 8, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000020000000238)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(0))), "r"(UINT64_C(0x0010001000000200)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(256))), "r"(UINT64_C(0x0010001000000210)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(512))), "r"(UINT64_C(0x0010001000000220)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(768))), "r"(UINT64_C(0x0010001000000230)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000240)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(1280))), "r"(UINT64_C(0x0010001000000250)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(1536))), "r"(UINT64_C(0x0010001000000260)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(1792))), "r"(UINT64_C(0x0010001000000270)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000280)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(2304))), "r"(UINT64_C(0x0010001000000290)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(2560))), "r"(UINT64_C(0x00100010000002a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(2816))), "r"(UINT64_C(0x00100010000002b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(3072))), "r"(UINT64_C(0x00100010000002c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(3328))), "r"(UINT64_C(0x00100010000002d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(3584))), "r"(UINT64_C(0x00100010000002e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(3840))), "r"(UINT64_C(0x00100010000002f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(4096))), "r"(UINT64_C(0x0010001000000300)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(4352))), "r"(UINT64_C(0x0010001000000310)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(4608))), "r"(UINT64_C(0x0010001000000320)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(4864))), "r"(UINT64_C(0x0010001000000330)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(5120))), "r"(UINT64_C(0x0010001000000340)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(5376))), "r"(UINT64_C(0x0010001000000350)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(5632))), "r"(UINT64_C(0x0010001000000360)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(5888))), "r"(UINT64_C(0x0010001000000370)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(6144))), "r"(UINT64_C(0x0010001000000380)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(6400))), "r"(UINT64_C(0x0010001000000390)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(6656))), "r"(UINT64_C(0x00100010000003a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(6912))), "r"(UINT64_C(0x00100010000003b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(7168))), "r"(UINT64_C(0x00100010000003c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(7424))), "r"(UINT64_C(0x00100010000003d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(7680))), "r"(UINT64_C(0x00100010000003e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(7936))), "r"(UINT64_C(0x00100010000003f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(8192))), "r"(UINT64_C(0x0010001000000400)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(8448))), "r"(UINT64_C(0x0010001000000410)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(8704))), "r"(UINT64_C(0x0010001000000420)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(8960))), "r"(UINT64_C(0x0010001000000430)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(9216))), "r"(UINT64_C(0x0010001000000440)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(9472))), "r"(UINT64_C(0x0010001000000450)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(9728))), "r"(UINT64_C(0x0010001000000460)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(9984))), "r"(UINT64_C(0x0010001000000470)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(10240))), "r"(UINT64_C(0x0010001000000480)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(10496))), "r"(UINT64_C(0x0010001000000490)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(10752))), "r"(UINT64_C(0x00100010000004a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(11008))), "r"(UINT64_C(0x00100010000004b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(11264))), "r"(UINT64_C(0x00100010000004c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(11520))), "r"(UINT64_C(0x00100010000004d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(11776))), "r"(UINT64_C(0x00100010000004e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(12032))), "r"(UINT64_C(0x00100010000004f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(12288))), "r"(UINT64_C(0x0010001000000500)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(12544))), "r"(UINT64_C(0x0010001000000510)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(12800))), "r"(UINT64_C(0x0010001000000520)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(13056))), "r"(UINT64_C(0x0010001000000530)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(13312))), "r"(UINT64_C(0x0010001000000540)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(13568))), "r"(UINT64_C(0x0010001000000550)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(13824))), "r"(UINT64_C(0x0010001000000560)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(14080))), "r"(UINT64_C(0x0010001000000570)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(14336))), "r"(UINT64_C(0x0010001000000580)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(14592))), "r"(UINT64_C(0x0010001000000590)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(14848))), "r"(UINT64_C(0x00100010000005a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(15104))), "r"(UINT64_C(0x00100010000005b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(15360))), "r"(UINT64_C(0x00100010000005c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(15616))), "r"(UINT64_C(0x00100010000005d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(15872))), "r"(UINT64_C(0x00100010000005e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(16128))), "r"(UINT64_C(0x00100010000005f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(16384))), "r"(UINT64_C(0x0010001000000600)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(16640))), "r"(UINT64_C(0x0010001000000610)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(16896))), "r"(UINT64_C(0x0010001000000620)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(17152))), "r"(UINT64_C(0x0010001000000630)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(17408))), "r"(UINT64_C(0x0010001000000640)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(17664))), "r"(UINT64_C(0x0010001000000650)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(17920))), "r"(UINT64_C(0x0010001000000660)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(18176))), "r"(UINT64_C(0x0010001000000670)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(18432))), "r"(UINT64_C(0x0010001000000680)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(18688))), "r"(UINT64_C(0x0010001000000690)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(18944))), "r"(UINT64_C(0x00100010000006a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(19200))), "r"(UINT64_C(0x00100010000006b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(19456))), "r"(UINT64_C(0x00100010000006c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(19712))), "r"(UINT64_C(0x00100010000006d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(19968))), "r"(UINT64_C(0x00100010000006e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(20224))), "r"(UINT64_C(0x00100010000006f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(20480))), "r"(UINT64_C(0x0010001000000700)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(20736))), "r"(UINT64_C(0x0010001000000710)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(20992))), "r"(UINT64_C(0x0010001000000720)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(21248))), "r"(UINT64_C(0x0010001000000730)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(21504))), "r"(UINT64_C(0x0010001000000740)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(21760))), "r"(UINT64_C(0x0010001000000750)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(22016))), "r"(UINT64_C(0x0010001000000760)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(22272))), "r"(UINT64_C(0x0010001000000770)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(22528))), "r"(UINT64_C(0x0010001000000780)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(22784))), "r"(UINT64_C(0x0010001000000790)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(23040))), "r"(UINT64_C(0x00100010000007a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(23296))), "r"(UINT64_C(0x00100010000007b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(23552))), "r"(UINT64_C(0x00100010000007c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(23808))), "r"(UINT64_C(0x00100010000007d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(24064))), "r"(UINT64_C(0x00100010000007e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(24320))), "r"(UINT64_C(0x00100010000007f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(24576))), "r"(UINT64_C(0x0010001000000800)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(24832))), "r"(UINT64_C(0x0010001000000810)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(25088))), "r"(UINT64_C(0x0010001000000820)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(25344))), "r"(UINT64_C(0x0010001000000830)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(25600))), "r"(UINT64_C(0x0010001000000840)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(25856))), "r"(UINT64_C(0x0010001000000850)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(26112))), "r"(UINT64_C(0x0010001000000860)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(26368))), "r"(UINT64_C(0x0010001000000870)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(26624))), "r"(UINT64_C(0x0010001000000880)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(26880))), "r"(UINT64_C(0x0010001000000890)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(27136))), "r"(UINT64_C(0x00100010000008a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(27392))), "r"(UINT64_C(0x00100010000008b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(27648))), "r"(UINT64_C(0x00100010000008c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(27904))), "r"(UINT64_C(0x00100010000008d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(28160))), "r"(UINT64_C(0x00100010000008e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(28416))), "r"(UINT64_C(0x00100010000008f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(28672))), "r"(UINT64_C(0x0010001000000900)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(28928))), "r"(UINT64_C(0x0010001000000910)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(29184))), "r"(UINT64_C(0x0010001000000920)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(29440))), "r"(UINT64_C(0x0010001000000930)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(29696))), "r"(UINT64_C(0x0010001000000940)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(29952))), "r"(UINT64_C(0x0010001000000950)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(30208))), "r"(UINT64_C(0x0010001000000960)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(30464))), "r"(UINT64_C(0x0010001000000970)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(30720))), "r"(UINT64_C(0x0010001000000980)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(30976))), "r"(UINT64_C(0x0010001000000990)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(31232))), "r"(UINT64_C(0x00100010000009a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(31488))), "r"(UINT64_C(0x00100010000009b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(31744))), "r"(UINT64_C(0x00100010000009c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(32000))), "r"(UINT64_C(0x00100010000009d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(32256))), "r"(UINT64_C(0x00100010000009e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)output_bf16 + UINT64_C(32512))), "r"(UINT64_C(0x00100010000009f0)) : "memory");
  __asm__ volatile ("fence" ::: "memory");
}

static uint32_t float_bits(float value) {
  union { float f; uint32_t u; } v = { .f = value };
  return v.u;
}

static float bf16_value(uint16_t bits) {
  union { uint32_t u; float f; } v = { .u = (uint32_t)bits << 16 };
  return v.f;
}

static float power_of_two(int exponent) {
  union { uint32_t u; float f; } v = {
    .u = exponent >= -126 ? (uint32_t)(exponent + 127) << 23 :
         (uint32_t)1 << (exponent + 149)
  };
  return v.f;
}

static int floor_log2_positive_bits(uint32_t bits) {
  int biased = (int)((bits >> 23) & 255);
  if (biased) return biased - 127;
  uint32_t fraction = bits & 0x7fffff;
  int leading = -1;
  while (fraction) { fraction >>= 1; ++leading; }
  return leading - 149;
}

static uint8_t source_fp6_code(uint16_t bits) {
  int sign = (bits & 0x8000) ? -1 : 1;
  int exponent = (bits >> 7) & 255;
  int mantissa = bits & 127;
  float value = 0.0f;
  if (exponent == 255) value = sign * 28.0f;
  else if (exponent != 0) {
    int e = exponent - 127;
    if (e >= -6 && e <= 7) {
      int high = (mantissa >> 5) & 3;
      int rounding = (mantissa >> 4) & 1;
      int sticky = (mantissa & 15) != 0;
      int rounded = high + (rounding & (sticky | (high & 1)));
      int out_e = e + (rounded >= 4);
      value = sign * (out_e > 7 ? 28.0f :
          (1.0f + (rounded >= 4 ? 0 : rounded) * 0.25f) * power_of_two(out_e));
    } else if (e == -7)
      value = sign * (mantissa <= 32 ? 2.0f :
                      mantissa <= 95 ? 3.0f : 4.0f) / 256.0f;
    else if (e == -8)
      value = sign * (mantissa < 64 ? 1.0f : 2.0f) / 256.0f;
    else if (e == -9)
      value = sign * (mantissa == 0 ? 0.0f : 1.0f) / 256.0f;
  }
  float magnitude = value < 0.0f ? -value : value;
  if (magnitude >= 32.0f) value = sign * 28.0f;
  else if (magnitude <= 0.0546875f) value = 0.0f;
  else if (magnitude >= 0.0625f && magnitude <= 0.21875f)
    value = sign * (magnitude <= 0.078125f ? 0.0625f :
                    magnitude <= 0.15625f ? 0.125f : 0.1875f);
  if (value == 0.0f) return 0;
  uint8_t code_sign = value < 0.0f ? 0x20 : 0;
  magnitude = value < 0.0f ? -value : value;
  if (magnitude < 0.25f) {
    int sub = (int)(magnitude / 0.0625f + 0.5f);
    return code_sign | (uint8_t)(sub > 3 ? 3 : sub);
  }
  int out_e = floor_log2_positive_bits(float_bits(magnitude));
  float base = power_of_two(out_e);
  int out_m = (int)((magnitude - base) / (base / 4.0f) + 0.5f);
  if (out_m >= 4) { out_m = 0; ++out_e; }
  int biased = out_e + 3;
  if (biased > 7) biased = 7;
  if (out_m > 3) out_m = 3;
  return code_sign | (uint8_t)((biased << 2) | out_m);
}

static uint8_t lut_code(uint32_t pair, uint32_t index, const uint8_t *output_lut) {
  const uint8_t *line = output_lut + pair * 12;
  uint32_t bit = index * 6, byte = bit >> 3;
  uint16_t chunk = line[byte];
  if (byte + 1 < 12) chunk |= (uint16_t)line[byte + 1] << 8;
  return (uint8_t)((chunk >> (bit & 7)) & 63);
}

static int fp6_fixed(uint8_t code) {
  int exponent = (code >> 2) & 7, mantissa = code & 3;
  if (exponent == 0 && mantissa == 0) return 0;
  int signed_e = exponent == 0 ? -2 : exponent - 3;
  int significand = (exponent == 0 ? 0 : 4) | mantissa;
  int magnitude = (significand << ((signed_e + 2) & 7)) & 255;
  return code & 0x20 ? -magnitude : magnitude;
}

static uint8_t nearest_lut_index(uint32_t pair, uint8_t code, const uint8_t *output_lut) {
  int input = fp6_fixed(code), best_distance = 0;
  uint8_t best = 0;
  for (uint8_t index = 0; index < 16; ++index) {
    int distance = input - fp6_fixed(lut_code(pair, index, output_lut));
    if (distance < 0) distance = -distance;
    distance &= 0x1ff;
    if (index == 0 || distance < best_distance) {
      best_distance = distance; best = index;
    }
  }
  return best;
}

static void radiance_header_requantize(const void *output_bf16, const uint8_t *output_lut, uint8_t *output_quantized, uint8_t *scratch_output_scales) {
  const uint16_t *input = (const uint16_t *)output_bf16;
  for (uint32_t row = 0; row < 128; ++row)
    for (uint32_t group = 0; group < 4; ++group) {
      uint32_t begin = row * 128 + group * 32;
      uint16_t maximum = 0;
      for (uint32_t offset = 0; offset < 32; ++offset) {
        uint16_t magnitude = input[begin + offset] & 0x7fff;
        if (magnitude > maximum) maximum = magnitude;
      }
      int exponent = maximum == 0 ? 0 :
          floor_log2_positive_bits((uint32_t)maximum << 16) - 4 + 127;
      uint8_t scale_code = maximum == 0 ? 0 :
          (uint8_t)(exponent < 0 ? 0 : exponent > 254 ? 254 : exponent);
      scratch_output_scales[row * 4 + group] = scale_code;
      float scale = power_of_two((int)scale_code - 127);
      for (uint32_t offset = 0; offset < 32; ++offset) {
        uint32_t col = group * 32 + offset;
        float scaled = bf16_value(input[begin + offset]) / scale;
        uint32_t bits = float_bits(scaled);
        uint16_t rounded = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        uint8_t code = source_fp6_code(rounded);
        uint8_t index = nearest_lut_index(row >> 1, code, output_lut);
        output_quantized[(row >> 1) * 128 + col] |=
            (uint8_t)(index << ((row & 1) ? 4 : 0));
      }
    }
}

void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, void *output_bf16, const void *output_lut, void *output_quantized, void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales) {
  mx_issue_commands(activation, activation_lut, activation_scales, output_bf16, output_lut, output_quantized, scratch_output_scales, weight, weight_lut, weight_scales);
  volatile uint8_t *quant = (volatile uint8_t *)output_quantized;
  for (uint32_t i = 0; i < 8192; ++i) quant[i] = 0;
  radiance_header_requantize(output_bf16, (const uint8_t *)output_lut, (uint8_t *)output_quantized, (uint8_t *)scratch_output_scales);
}
