# Full SmolVLA session: compiler versus Nicolas MX matrix issuers

The captured SmolVLA session contains one prefix encode step, ten recurrent
flow denoise steps, and one action decode step. Both RV64 executables ran on
the same Nicolas `e4m3_vpu` Spike extension with `mem.host_tracking=0`.
The source variant changes only the MX matrix issuer catalogs for prefix and
flow. The compiled host graph, state adapters, VPU objects, profile,
quantization, checkpoint weights, and action stage remain fixed. The source
issuers call Nicolas's `mx_native.h` at npu-exploration revision `0344bcb`.

| Stage | Compiler cycles | Nicolas issuer cycles | Difference |
| --- | ---: | ---: | ---: |
| Prefix encode | 50,739,160,346 | 50,557,656,572 | +181,503,774 |
| Flow denoise, ten steps | 21,327,379,772 | 21,318,022,826 | +9,356,946 |
| Action decode | 4,236 | 4,236 | 0 |
| **Full 12-step session** | **72,066,544,354** | **71,875,683,634** | **+190,860,720 (+0.266%)** |

The [comparison receipt](comparison.json) confirms all 12 steps, all 33
output digests, the same captured work and weight digest, the same Spike
binary and extension, and the same timing settings. The compiler takes
0.266% more modeled cycles in this paired experiment. The full session
completed; independent quantized model accuracy remains unqualified.

The copied [compiler](compiler/mx_session_run.json) and
[Nicolas issuer](nicolas/mx_session_run.json) run receipts include the
matching raw Spike logs. The [link receipts](compiler_link.json) and
[source link receipt](nicolas_link.json) record the stage object hashes.
They retain absolute build paths from the measured task; the small receipts
and logs are archived here, while the multi-gigabyte ELFs remain local.

To recheck the archived pair with the same receipt validator:

```sh
python -m tools.compare_linked_session_cycles \
  --reference-link docs/evidence/nicolas_matrix_substitution_smolvla_full_20261013/compiler_link.json \
  --reference-run docs/evidence/nicolas_matrix_substitution_smolvla_full_20261013/compiler/mx_session_run.json \
  --candidate-link docs/evidence/nicolas_matrix_substitution_smolvla_full_20261013/nicolas_link.json \
  --candidate-run docs/evidence/nicolas_matrix_substitution_smolvla_full_20261013/nicolas/mx_session_run.json \
  --out "$TMPDIR/smolvla_full_comparison_recheck.json"
```

This measures a complete model session with native matrix issuers. It is
not a full handwritten fused MX+VPU implementation of the captured graph.
