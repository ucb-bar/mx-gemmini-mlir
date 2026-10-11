#ifndef MX_ISSUE_H
#define MX_ISSUE_H

/* Buffer order, sizes, and layouts are in object_manifest.json. */
void mx_issue(const void *activation, const void *activation_lut, const void *activation_scales, const void *output_lut, const void *output_quantized, const void *scratch_output_scales, const void *weight, const void *weight_lut, const void *weight_scales);

#endif
