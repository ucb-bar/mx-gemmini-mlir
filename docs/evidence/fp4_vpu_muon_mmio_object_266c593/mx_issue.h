#ifndef MX_ISSUE_H
#define MX_ISSUE_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Buffer order and sizes are in object_manifest.json. */
void mx_issue(const void *activation, const void *activation_scales, const void *output_bf16, const void *scratch_output_scales, const void *weight, const void *weight_scales, uintptr_t mx_control_base);

#ifdef __cplusplus
}
#endif

#endif
