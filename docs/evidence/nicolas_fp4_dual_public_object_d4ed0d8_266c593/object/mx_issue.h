#ifndef MX_ISSUE_H
#define MX_ISSUE_H

/* Buffer order and sizes are in object_manifest.json. */
void mx_issue(const void *X, const void *scales_hw, const void *scales_hw2, const void *codes_flat_hw, const void *codes_tiled_hw);

#endif
