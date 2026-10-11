#ifndef MX_ISSUE_H
#define MX_ISSUE_H

/* Buffer order and sizes are in object_manifest.json. */
void mx_issue(const void *Q_SCALES, const void *KT_SCALES, const void *V_SCALES, const void *Q_IN, const void *KT_IN, const void *V_IN, const void *scale_sink, const void *S_hw, const void *p_scales, const void *O_hw, const void *P_hw, const void *p_codes_hw);

#endif
