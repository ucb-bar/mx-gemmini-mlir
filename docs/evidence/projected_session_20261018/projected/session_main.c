/* MX target's Spike entry; state scheduling is Merlin-generated. */
#include <stddef.h>
#include <stdint.h>
#include "merlin_session.h"
#include "output_sha256.h"
void console_init(void);
void htif_puts(const char *);
void htif_putd(long);
void htif_puthex(unsigned long long);
void htif_putc(char);
void htif_exit(int);
void merlin_arena_reset(void);
unsigned long long merlin_memref_rank_mismatches(void);
void *merlin_stage_0_output(int);
size_t merlin_stage_0_output_bytes(int);
void *merlin_stage_1_output(int);
size_t merlin_stage_1_output_bytes(int);
static uint64_t read_cycle(void) {
  uint64_t value;
  __asm__ volatile ("rdcycle %0" : "=r" (value) : : "memory");
  return value;
}
static uint64_t read_instret(void) {
  uint64_t value;
  __asm__ volatile ("rdinstret %0" : "=r" (value) : : "memory");
  return value;
}

static void report_preparation(int program, uint64_t start_cycles, uint64_t start_instructions,
                               uint64_t *total_cycles, uint64_t *total_instructions) {
  uint64_t instructions = read_instret() - start_instructions;
  uint64_t cycles = read_cycle() - start_cycles;
  *total_cycles += cycles; *total_instructions += instructions;
  htif_puts("SESSION_PREP_PERF "); htif_putd(program); htif_putc(' ');
  htif_putd((long)cycles); htif_putc(' '); htif_putd((long)instructions); htif_putc('\n');
}

static const int OUTPUT_COUNTS[MERLIN_SESSION_N_PROGRAMS] = {2, 2};
static void *output_pointer(int program, int output) {
  switch (program) {
    case 0: return merlin_stage_0_output(output);
    case 1: return merlin_stage_1_output(output);
    default: return 0;
  }
}
static size_t output_bytes(int program, int output) {
  switch (program) {
    case 0: return merlin_stage_0_output_bytes(output);
    case 1: return merlin_stage_1_output_bytes(output);
    default: return 0;
  }
}
static void report(int program, long step, int output, const void *value, size_t bytes) {
  static const char hex[] = "0123456789abcdef";
  uint8_t hash[32];
  merlin_output_sha256_raw(value, bytes, hash);
  htif_puts("SESSION_SHA "); htif_putd(program); htif_putc(' ');
  htif_putd(step); htif_putc(' '); htif_putd(output); htif_putc(' ');
  htif_putd((long)bytes); htif_putc(' ');
  for (int i = 0; i < 32; ++i) {
    htif_putc(hex[hash[i] >> 4]); htif_putc(hex[hash[i] & 15]);
  }
  htif_putc('\n');
}
static void dump_output(int program, long step, int output,
                        const void *value, size_t bytes) {
  static const char hex[] = "0123456789abcdef";
  const uint8_t *data = (const uint8_t *)value;
  htif_puts("SESSION_HEX "); htif_putd(program); htif_putc(' ');
  htif_putd(step); htif_putc(' '); htif_putd(output); htif_putc(' ');
  htif_putd((long)bytes); htif_putc(' ');
  for (size_t i = 0; i < bytes; ++i) {
    htif_putc(hex[data[i] >> 4]); htif_putc(hex[data[i] & 15]);
  }
  htif_putc('\n');
}
int main(int hart) {
  if (hart != 0) for (;;) ;
  console_init();
  const void *weights = (const void *)0x200000000ULL;
  uint64_t preparation_total_cycles = 0, preparation_total_instructions = 0;
  uint64_t preparation_start_cycles = read_cycle(), preparation_start_instructions = read_instret();
  merlin_session_reset();
  report_preparation(-1, preparation_start_cycles, preparation_start_instructions,
                      &preparation_total_cycles, &preparation_total_instructions);
  uint64_t total_cycles = 0, total_instructions = 0;
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; ++program) {
    preparation_start_cycles = read_cycle();
    preparation_start_instructions = read_instret();
    int preparation_status = merlin_session_prepare_program(program);
    report_preparation(program, preparation_start_cycles, preparation_start_instructions,
                        &preparation_total_cycles, &preparation_total_instructions);
    if (preparation_status != 0) {
      htif_puts("FAIL session binding\n"); htif_exit(1); return 1;
    }
    long steps = merlin_session_program_steps(program);
    for (long step = 0; step < steps; ++step) {
      htif_puts("SESSION_BEGIN "); htif_putd(program); htif_putc(' ');
      htif_putd(step); htif_putc('\n');
      uint64_t start_cycles = read_cycle();
      uint64_t start_instructions = read_instret();
      merlin_arena_reset();
      if (merlin_session_run_step(program, weights, step, 0) != 0) {
        htif_puts("FAIL session state\n"); htif_exit(1); return 1;
      }
      uint64_t instructions = read_instret() - start_instructions;
      uint64_t cycles = read_cycle() - start_cycles;
      total_cycles += cycles;
      total_instructions += instructions;
      htif_puts("SESSION_PERF "); htif_putd(program); htif_putc(' ');
      htif_putd(step); htif_putc(' '); htif_putd((long)cycles);
      htif_putc(' '); htif_putd((long)instructions); htif_putc('\n');
      for (int output = 0; output < OUTPUT_COUNTS[program]; ++output) {
        void *value = output_pointer(program, output);
        size_t bytes = output_bytes(program, output);
        if (!value || !bytes) {
          htif_puts("FAIL session output ABI program="); htif_putd(program);
          htif_puts(" step="); htif_putd(step);
          htif_puts(" output="); htif_putd(output);
          htif_puts(" pointer="); htif_puthex((unsigned long long)(uintptr_t)value);
          htif_puts(" bytes="); htif_putd((long)bytes);
          htif_putc('\n'); htif_exit(1); return 1;
        }
        report(program, step, output, value, bytes);
        if ((program == 0 && output == 0) || (program == 0 && output == 1) || (program == 1 && output == 0) || (program == 1 && output == 1)) dump_output(program, step, output, value, bytes);
      }
    }
  }
  htif_puts("SESSION_TOTAL "); htif_putd((long)total_cycles);
  htif_putc(' '); htif_putd((long)total_instructions); htif_putc('\n');
  htif_puts("SESSION_EXECUTION_TOTAL ");
  htif_putd((long)(total_cycles + preparation_total_cycles)); htif_putc(' ');
  htif_putd((long)(total_instructions + preparation_total_instructions)); htif_putc('\n');
  htif_puts("METRIC memref_rank_mismatch ");
  htif_putd((long)merlin_memref_rank_mismatches()); htif_putc('\n');
  htif_puts("DONE\n"); htif_exit(0); return 0;
}
