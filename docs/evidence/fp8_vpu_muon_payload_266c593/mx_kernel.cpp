#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include "mx_issue.h"

extern "C" {
extern const uint8_t activation[];
extern const uint8_t activation_scales[];
extern const uint8_t weight[];
extern const uint8_t weight_scales[];
extern const uint8_t mx_expected_bf16[];
uint8_t output_bf16[131072] __attribute__((aligned(64)));
uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
}
volatile uint32_t mx_mismatch_count = UINT32_MAX;
volatile uint32_t mx_completed = 0;
static void worker(void *, uint32_t lane, uint32_t, uint32_t block) {
  if (lane != 0 || block != 0) return;
  mx_issue(activation, activation_scales, output_bf16, scratch_output_scales, weight, weight_scales, UINT32_C(0x00084000));
  mu_fence();
  const uint16_t *got = reinterpret_cast<const uint16_t *>(output_bf16);
  const uint16_t *expected = reinterpret_cast<const uint16_t *>(mx_expected_bf16);
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < 65536; ++i)
    mismatches += got[i] != expected[i];
  mx_mismatch_count = mismatches;
  mu_fence();
  mx_completed = 1;
  mu_fence();
}
extern "C" int main() {
  mu_schedule(worker, nullptr, 1);
  return 0;
}
