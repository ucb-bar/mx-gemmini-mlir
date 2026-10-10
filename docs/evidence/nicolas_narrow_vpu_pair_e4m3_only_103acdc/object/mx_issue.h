#ifndef MX_ISSUE_H
#define MX_ISSUE_H

/* Buffer order, sizes, and layouts are in object_manifest.json. */
void mx_issue(const void *a1_activation, const void *a1_scales, const void *b1_scales, const void *b1_weight, const void *b2_scales, const void *b2_weight, const void *c1_bf16_observed, const void *c1_scales, const void *c1_tiled, const void *c2_scales, const void *c2_tiled);

#endif
