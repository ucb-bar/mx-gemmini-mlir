#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_setup(void) {
  __asm__ volatile (".insn r 0x7b, 3, 7, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000000)), "r"(UINT64_C(0x0000000000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x000000000001c004)), "r"(UINT64_C(0x0001000000000000)) : "memory");
}

#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_a64(const void *src) {
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(0))), "r"(UINT64_C(0x0010004000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(64))), "r"(UINT64_C(0x0010004000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2048))), "r"(UINT64_C(0x0010004000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2112))), "r"(UINT64_C(0x00100040000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4096))), "r"(UINT64_C(0x0010004000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4160))), "r"(UINT64_C(0x0010004000000140)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6144))), "r"(UINT64_C(0x0010004000000180)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6208))), "r"(UINT64_C(0x00100040000001c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8192))), "r"(UINT64_C(0x0010004000000200)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8256))), "r"(UINT64_C(0x0010004000000240)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10240))), "r"(UINT64_C(0x0010004000000280)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10304))), "r"(UINT64_C(0x00100040000002c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12288))), "r"(UINT64_C(0x0010004000000300)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12352))), "r"(UINT64_C(0x0010004000000340)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14336))), "r"(UINT64_C(0x0010004000000380)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14400))), "r"(UINT64_C(0x00100040000003c0)) : "memory");
}

#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_b16(const void *src) {
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000100101)), "r"(UINT64_C(0x0000000000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(0))), "r"(UINT64_C(0x0010001000000400)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(16))), "r"(UINT64_C(0x0010001000000410)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(32))), "r"(UINT64_C(0x0010001000000420)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(48))), "r"(UINT64_C(0x0010001000000430)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(64))), "r"(UINT64_C(0x0010001000000440)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(80))), "r"(UINT64_C(0x0010001000000450)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(96))), "r"(UINT64_C(0x0010001000000460)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(112))), "r"(UINT64_C(0x0010001000000470)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000480)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2064))), "r"(UINT64_C(0x0010001000000490)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2080))), "r"(UINT64_C(0x00100010000004a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2096))), "r"(UINT64_C(0x00100010000004b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2112))), "r"(UINT64_C(0x00100010000004c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2128))), "r"(UINT64_C(0x00100010000004d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2144))), "r"(UINT64_C(0x00100010000004e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(2160))), "r"(UINT64_C(0x00100010000004f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4096))), "r"(UINT64_C(0x0010001000000500)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4112))), "r"(UINT64_C(0x0010001000000510)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4128))), "r"(UINT64_C(0x0010001000000520)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4144))), "r"(UINT64_C(0x0010001000000530)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4160))), "r"(UINT64_C(0x0010001000000540)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4176))), "r"(UINT64_C(0x0010001000000550)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4192))), "r"(UINT64_C(0x0010001000000560)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(4208))), "r"(UINT64_C(0x0010001000000570)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6144))), "r"(UINT64_C(0x0010001000000580)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6160))), "r"(UINT64_C(0x0010001000000590)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6176))), "r"(UINT64_C(0x00100010000005a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6192))), "r"(UINT64_C(0x00100010000005b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6208))), "r"(UINT64_C(0x00100010000005c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6224))), "r"(UINT64_C(0x00100010000005d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6240))), "r"(UINT64_C(0x00100010000005e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(6256))), "r"(UINT64_C(0x00100010000005f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8192))), "r"(UINT64_C(0x0010001000000600)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8208))), "r"(UINT64_C(0x0010001000000610)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8224))), "r"(UINT64_C(0x0010001000000620)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8240))), "r"(UINT64_C(0x0010001000000630)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8256))), "r"(UINT64_C(0x0010001000000640)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8272))), "r"(UINT64_C(0x0010001000000650)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8288))), "r"(UINT64_C(0x0010001000000660)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(8304))), "r"(UINT64_C(0x0010001000000670)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10240))), "r"(UINT64_C(0x0010001000000680)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10256))), "r"(UINT64_C(0x0010001000000690)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10272))), "r"(UINT64_C(0x00100010000006a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10288))), "r"(UINT64_C(0x00100010000006b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10304))), "r"(UINT64_C(0x00100010000006c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10320))), "r"(UINT64_C(0x00100010000006d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10336))), "r"(UINT64_C(0x00100010000006e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(10352))), "r"(UINT64_C(0x00100010000006f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12288))), "r"(UINT64_C(0x0010001000000700)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12304))), "r"(UINT64_C(0x0010001000000710)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12320))), "r"(UINT64_C(0x0010001000000720)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12336))), "r"(UINT64_C(0x0010001000000730)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12352))), "r"(UINT64_C(0x0010001000000740)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12368))), "r"(UINT64_C(0x0010001000000750)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12384))), "r"(UINT64_C(0x0010001000000760)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(12400))), "r"(UINT64_C(0x0010001000000770)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14336))), "r"(UINT64_C(0x0010001000000780)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14352))), "r"(UINT64_C(0x0010001000000790)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14368))), "r"(UINT64_C(0x00100010000007a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14384))), "r"(UINT64_C(0x00100010000007b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14400))), "r"(UINT64_C(0x00100010000007c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14416))), "r"(UINT64_C(0x00100010000007d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14432))), "r"(UINT64_C(0x00100010000007e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 2, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(14448))), "r"(UINT64_C(0x00100010000007f0)) : "memory");
}

#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_scale(const void *src) {
  __asm__ volatile (".insn r 0x7b, 3, 27, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)src + UINT64_C(0))), "r"(UINT64_C(0x0000000000000200)) : "memory");
}

#include <stdint.h>

_Static_assert(sizeof(uintptr_t) == 8, "MX Rocket commands require RV64");

void mx_issue_mvout(const void *dst) {
  __asm__ volatile (".insn r 0x7b, 3, 0, x0, %0, %1" : : "r"(UINT64_C(0x0000000000000002)), "r"(UINT64_C(0x0000000000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(0))), "r"(UINT64_C(0x0010001000000000)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(256))), "r"(UINT64_C(0x0010001000000010)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(512))), "r"(UINT64_C(0x0010001000000020)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(768))), "r"(UINT64_C(0x0010001000000030)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(1024))), "r"(UINT64_C(0x0010001000000040)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(1280))), "r"(UINT64_C(0x0010001000000050)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(1536))), "r"(UINT64_C(0x0010001000000060)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(1792))), "r"(UINT64_C(0x0010001000000070)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(2048))), "r"(UINT64_C(0x0010001000000080)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(2304))), "r"(UINT64_C(0x0010001000000090)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(2560))), "r"(UINT64_C(0x00100010000000a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(2816))), "r"(UINT64_C(0x00100010000000b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(3072))), "r"(UINT64_C(0x00100010000000c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(3328))), "r"(UINT64_C(0x00100010000000d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(3584))), "r"(UINT64_C(0x00100010000000e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(3840))), "r"(UINT64_C(0x00100010000000f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(4096))), "r"(UINT64_C(0x0010001000000100)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(4352))), "r"(UINT64_C(0x0010001000000110)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(4608))), "r"(UINT64_C(0x0010001000000120)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(4864))), "r"(UINT64_C(0x0010001000000130)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(5120))), "r"(UINT64_C(0x0010001000000140)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(5376))), "r"(UINT64_C(0x0010001000000150)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(5632))), "r"(UINT64_C(0x0010001000000160)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(5888))), "r"(UINT64_C(0x0010001000000170)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(6144))), "r"(UINT64_C(0x0010001000000180)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(6400))), "r"(UINT64_C(0x0010001000000190)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(6656))), "r"(UINT64_C(0x00100010000001a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(6912))), "r"(UINT64_C(0x00100010000001b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(7168))), "r"(UINT64_C(0x00100010000001c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(7424))), "r"(UINT64_C(0x00100010000001d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(7680))), "r"(UINT64_C(0x00100010000001e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(7936))), "r"(UINT64_C(0x00100010000001f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(8192))), "r"(UINT64_C(0x0010001000000200)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(8448))), "r"(UINT64_C(0x0010001000000210)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(8704))), "r"(UINT64_C(0x0010001000000220)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(8960))), "r"(UINT64_C(0x0010001000000230)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(9216))), "r"(UINT64_C(0x0010001000000240)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(9472))), "r"(UINT64_C(0x0010001000000250)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(9728))), "r"(UINT64_C(0x0010001000000260)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(9984))), "r"(UINT64_C(0x0010001000000270)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(10240))), "r"(UINT64_C(0x0010001000000280)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(10496))), "r"(UINT64_C(0x0010001000000290)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(10752))), "r"(UINT64_C(0x00100010000002a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(11008))), "r"(UINT64_C(0x00100010000002b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(11264))), "r"(UINT64_C(0x00100010000002c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(11520))), "r"(UINT64_C(0x00100010000002d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(11776))), "r"(UINT64_C(0x00100010000002e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(12032))), "r"(UINT64_C(0x00100010000002f0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(12288))), "r"(UINT64_C(0x0010001000000300)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(12544))), "r"(UINT64_C(0x0010001000000310)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(12800))), "r"(UINT64_C(0x0010001000000320)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(13056))), "r"(UINT64_C(0x0010001000000330)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(13312))), "r"(UINT64_C(0x0010001000000340)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(13568))), "r"(UINT64_C(0x0010001000000350)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(13824))), "r"(UINT64_C(0x0010001000000360)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(14080))), "r"(UINT64_C(0x0010001000000370)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(14336))), "r"(UINT64_C(0x0010001000000380)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(14592))), "r"(UINT64_C(0x0010001000000390)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(14848))), "r"(UINT64_C(0x00100010000003a0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(15104))), "r"(UINT64_C(0x00100010000003b0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(15360))), "r"(UINT64_C(0x00100010000003c0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(15616))), "r"(UINT64_C(0x00100010000003d0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(15872))), "r"(UINT64_C(0x00100010000003e0)) : "memory");
  __asm__ volatile (".insn r 0x7b, 3, 3, x0, %0, %1" : : "r"(((uint64_t)(uintptr_t)dst + UINT64_C(16128))), "r"(UINT64_C(0x00100010000003f0)) : "memory");
}
