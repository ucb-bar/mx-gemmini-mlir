/* Matched kernel calls; packing and output checks are outside both timers. */
#include "include/gemmini.h"
#include "pair_config.h"
#include <stdint.h>
#include <stdio.h>

extern const unsigned char payload[];
void native_issue(const void *, const void *, const void *, const void *,
                  const void *, const void *);
void compiler_issue(const void *, const void *, const void *, const void *,
                    const void *, const void *);
typedef void (*issuer)(const void *, const void *, const void *, const void *,
                       const void *, const void *);
static uint16_t native_output[PAIR_WORDS + 64] __attribute__((aligned(64)));
static uint16_t compiler_output[PAIR_WORDS + 64] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));

static inline uint64_t cycle(void) {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

__attribute__((noinline)) static uint64_t measure(issuer call,
                                                  uint16_t *output) {
  gemmini_flush(0);
  gemmini_fence();
  uint64_t begin = cycle();
  call(payload + OFFSET_A, payload + OFFSET_AS, payload + OFFSET_B,
       payload + OFFSET_BS, output, scratch);
  gemmini_fence();
  return cycle() - begin;
}

static void dump(const char *role, unsigned trial, const uint16_t *output) {
  for (unsigned offset = 0; offset < PAIR_WORDS; offset += 32) {
    unsigned count = PAIR_WORDS - offset < 32 ? PAIR_WORDS - offset : 32;
    printf("PAIR_WORDS %s %u %u %u", role, trial, offset, count);
    for (unsigned i = 0; i < count; ++i)
      printf(" %04x", output[offset + i]);
    printf("\n");
  }
}

int main(void) {
  const uint16_t *gold = (const uint16_t *)(payload + OFFSET_GOLD);
  for (unsigned trial = 0; trial < PAIR_TRIALS; ++trial) {
    for (unsigned i = 0; i < PAIR_WORDS + 64; ++i)
      native_output[i] = compiler_output[i] = 0xa55a;
    uint64_t native_cycles, compiler_cycles;
    if ((trial + PAIR_COMPILER_FIRST) & 1) {
      compiler_cycles = measure(compiler_issue, compiler_output + 32);
      native_cycles = measure(native_issue, native_output + 32);
    } else {
      native_cycles = measure(native_issue, native_output + 32);
      compiler_cycles = measure(compiler_issue, compiler_output + 32);
    }
    unsigned mismatch = 0, oracle = 0, guard = 0;
    for (unsigned i = 0; i < PAIR_WORDS; ++i) {
      mismatch += native_output[i + 32] != compiler_output[i + 32];
      oracle += native_output[i + 32] != gold[i] ||
                compiler_output[i + 32] != gold[i];
    }
    for (unsigned i = 0; i < 32; ++i)
      guard += native_output[i] != 0xa55a || compiler_output[i] != 0xa55a ||
               native_output[PAIR_WORDS + 32 + i] != 0xa55a ||
               compiler_output[PAIR_WORDS + 32 + i] != 0xa55a;
    printf("PAIR trial=%u native=%lu compiler=%lu mismatch=%u oracle=%u "
           "guard=%u\n",
           trial, (unsigned long)native_cycles, (unsigned long)compiler_cycles,
           mismatch, oracle, guard);
    dump("native", trial, native_output + 32);
    dump("compiler", trial, compiler_output + 32);
    if (mismatch || oracle || guard)
      return 3;
  }
  printf("PAIR_DONE\n");
  return 0;
}
