#include <stdint.h>
#include <mu_schedule.h>
#include "mx_issue.h"

volatile uint32_t mx_probe_enable = 0;
static void worker(void *, uint32_t lane, uint32_t, uint32_t block) {
  if (lane == 0 && block == 0 && mx_probe_enable)
    mx_issue(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, UINT32_C(0x00084000));
}
extern "C" int main() {
  mu_schedule(worker, nullptr, 1);
  return 0;
}
