#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(uintptr_t) == 8, "MX native commands require RV64");
#ifdef MX_TRACE_COMMANDS
extern void mx_trace_command(uint32_t funct, uint64_t rs1, uint64_t rs2);
extern void mx_trace_fence(void);
#define MX_CMD(F, A, B) mx_trace_command((F), (uint64_t)(A), (uint64_t)(B))
#define MX_FENCE() mx_trace_fence()
#else
#define MX_CMD(F, A, B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" \
    : : "r"((uint64_t)(A)), "r"((uint64_t)(B)) : "memory")
#define MX_FENCE() __asm__ volatile ("fence" ::: "memory")
#endif
static void mx_issue_encoded_scheduled(uint8_t *activation, uint8_t *activation_scales, uint8_t *weight, uint8_t *weight_scales, uint16_t *output_bf16, uint8_t *scratch_output_scales) {
    MX_CMD(7, 0, 0);
    MX_CMD(0, UINT64_C(124932), UINT64_C(1) << 48);
    MX_CMD(0, UINT64_C(1048833), UINT64_C(128));
    MX_CMD(0, UINT64_C(1048841), UINT64_C(176));
    MX_CMD(0, 2, UINT64_C(1408));
    for (int chunk = 0; chunk < 11; chunk++) {
        for (int wave = 0; wave < 1; wave++) {
            do {
      MX_CMD(31, (uint64_t)(uintptr_t)((uint8_t *)activation_scales + (wave * 128)), (uint64_t)(uintptr_t)((uint8_t *)weight_scales + ((wave * 1408) + (chunk * 32))));
      MX_CMD(32, UINT64_C(32), UINT64_C(352));
      MX_CMD(9, 0, UINT64_C(34359803905));
      MX_CMD(10, (uint64_t)(uintptr_t)((uint8_t *)activation + (wave * 128)), (uint64_t)(uintptr_t)((uint8_t *)weight + ((wave * 22528) + (chunk * 16))));
      MX_CMD(11, 0, (((wave == 0) ? 1 : 0)) ? (uint64_t)(uintptr_t)((uint8_t *)output_bf16 + (chunk * 64)) : 0);
      MX_CMD(12, UINT64_C(128), UINT64_C(176));
      MX_CMD(13, 0, UINT64_C(352));
      MX_CMD(8, (UINT64_C(1) << 18) | ((uint64_t)((((((chunk + wave) % 2) + 2) % 2) + 1)) << 16) | (uint64_t)(((wave == 0) ? 0 : 1)), 0);
    } while (0);
        }
    }
    MX_FENCE();
}

void mx_issue_encoded(const void *activation, const void *activation_scales,
              const void *weight, const void *weight_scales,
              const void *output_bf16, const void *scratch_output_scales) {
  if (!activation || !activation_scales || !weight || !weight_scales ||
      !output_bf16 || !scratch_output_scales ||
      (((uintptr_t)activation_scales | (uintptr_t)weight_scales) & 7)) __builtin_trap();
  mx_issue_encoded_scheduled((uint8_t *)activation, (uint8_t *)activation_scales,
                     (uint8_t *)weight, (uint8_t *)weight_scales,
                     (uint16_t *)output_bf16, (uint8_t *)scratch_output_scales);
}
