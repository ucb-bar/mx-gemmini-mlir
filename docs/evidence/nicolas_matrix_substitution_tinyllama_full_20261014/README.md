<a id="full-tinyllama-session-compiler-versus-nicolas-mx-matrix-issuers"></a>

# Full TinyLlama session: compiler versus reference MX matrix issuers

The captured TinyLlama session includes one prefill and 32 recurrent decode
steps. Both RV64 executables ran with the same reference `e4m3_vpu` Spike
extension and `mem.host_tracking=0`. The source variant replaces only the MX
matrix issuer catalogs with calls to the reference implementation’s
[`mx_native.h`](https://github.com/ucb-bar/npu-exploration/blob/0344bcba70b899039ccd0ac0801ca5605d46d3eb/baremetal/mxgemmini/include/mx_native.h)
at npu-exploration revision `0344bcb`. The host graph, VPU objects, profile,
checkpoint weights, state handoff, and result checks remain fixed.

| Stage | Compiler cycles | reference issuer cycles | Compiler difference |
| --- | ---: | ---: | ---: |
| Prefill (1 step) | 11,978,634,706 | 11,974,767,715 | +3,866,991 |
| Decode (32 steps) | 71,568,929,647 | 71,569,835,080 | −905,433 |
| **Full 33-step session** | **83,547,564,353** | **83,544,602,795** | **+2,961,558 (+0.004%)** |

The [comparison receipt](comparison.json) verifies all 33 steps and 99
output digests, shared captured work and weights, identical Spike binary and
extension, and identical timing settings. Independent quantized model
accuracy remains unqualified.

The [compiler](compiler/mx_session_run.json) and
[reference issuer](nicolas/mx_session_run.json) run receipts have matching raw
Spike logs. Their [link receipts](compiler_link.json) and
[source link receipt](nicolas_link.json) retain the measured object hashes.
The [prefill](nicolas_prefill_catalog_receipt.json) and
[decode](nicolas_decode_catalog_receipt.json) source catalog receipts bind
both replacement objects to the reference implementation’s pinned source revision and header.
They retain absolute paths from the measured task. The small receipts and
logs are archived here, while the multi-gigabyte ELFs stay local.

To recheck the archived pair with the same receipt validator:

```sh
python -m tools.compare_linked_session_cycles \
  --reference-link docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/compiler_link.json \
  --reference-run docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/compiler/mx_session_run.json \
  --candidate-link docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/nicolas_link.json \
  --candidate-run docs/evidence/nicolas_matrix_substitution_tinyllama_full_20261014/nicolas/mx_session_run.json \
  --out "$TMPDIR/tinyllama_full_comparison_recheck.json"
```

This is an equal-input matrix issuer substitution, not a fused handwritten
MX+VPU implementation of the captured graph.
