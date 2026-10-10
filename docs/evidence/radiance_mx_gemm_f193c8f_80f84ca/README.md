# Current-source 31-driver MX GEMM replay

At compiler `f193c8f`, `python -m tools.reproduce_radiance_roster` regenerated
all 31 MX GEMM drivers in the Radiance `spatter-workloads` checkout at
`80f84ca`. It generated missing source headers, recaptured each operation
through model2MLIR `e9ded36`, compiled a standalone RV64 program, and ran
Nicolas's pinned Spike extension. All 23 BF16-output and eight quantized-output
drivers matched their source goldens. The checker compared 466,944 output
values or packed code bytes and 3,328 E8M0 scales. The command also compared
frontend, physical, object, ELF, and Spike-log hashes to the earlier pinned
baseline before writing `reproduction.json`.

This compact archive contains the final receipt, header/front-end/Spike
indices, 31 frontend receipts, 31 simulator receipts, and their 31 Spike logs.
The generated ELF and MLIR files remain in the local run directory; their
hashes are recorded in the receipts and checked against the existing baseline.
The source drivers include recipes excluded from Radiance's Makefile, so this
is compiler-on-Spike source parity rather than a claim that all 31 source ELFs
are built by Radiance's own default target.

The replay command and its `--compatible-source-revision` gate are documented
in [the compiled MX pipeline](../../compiled_mx_pipeline.md#latest-complete-radiance-mx-gemm-roster).
