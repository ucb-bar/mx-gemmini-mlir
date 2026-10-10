#ifndef MX_ISSUE_H
#define MX_ISSUE_H

/* See object_manifest.json for buffer byte lengths and layouts. */
void mx_issue(const void *a1_activation, const void *a1_scales, const void *b1_scales, const void *b1_weight, const void *b2_scales, const void *b2_weight, const void *c1_bf16_observed, const void *c1_scales_0, const void *c1_scales_1, const void *c1_scales_mm1, const void *c1_tiled_0, const void *c1_tiled_1, const void *c2_scales_0, const void *c2_scales_1, const void *c2_tiled_0, const void *c2_tiled_1);
#endif
