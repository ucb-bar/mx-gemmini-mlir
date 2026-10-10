# Radiance host requantization objects on Nicolas's Spike

The eight `gemm_mxgemmini/*.requant.cpp` cases in the Radiance `80f84ca`
source roster were captured with model2MLIR `e9ded36`, bound to Nicolas's
MX RTL profiles at `266c593`, and compiled from typed MX MLIR into data-free
RV64 RoCC objects at compiler revision `ddffa18`. The native MX verifier
accepted all eight bound graphs. Each object takes runtime input, BF16,
quantized-output, and scale pointers. The FP6 objects also take the complete
output LUT through the pointer ABI.

Nicolas's pinned Spike model matched **90,112 quantized code bytes** and
**3,328 E8M0 scales** against the source goldens. The result repeated from a
fresh checkout of the published `handwritten-implementation` branch: the
two [`index.json`](index.json) files and every archived per-case object,
ELF, manifest, generated C file, and Spike log were byte-identical. The
[`evidence.json`](evidence.json) file records every raw artifact digest.
Compressed files use deterministic gzip; their recorded digest refers to
the decompressed content.

A second replay used a new shallow clone at published revision `6975623`.
It configured and rebuilt `mx-gemmini-opt` from source, reproduced the native
verifier's hash, and generated a byte-identical eight-case index. The
[clone receipt](published_clone_replay.json), [configure log](published_clone_configure.log),
and [build log](published_clone_build.log) record that check.

The source roster is produced by `tools.reproduce_radiance_roster`. Given its
`spike/` directory, replay and archive these objects with:

```sh
python -m tools.qualify_host_requant_roster \
  --source-root "$MX_REPRODUCTION/spike" \
  --profiles profiles/gemmini-mx-cleanup-266c593 \
  --rtl-root "$MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --mx-opt build/tools/mx-gemmini-opt \
  --out-dir /tmp/mx-host-object-run --workers 2
python -m tools.archive_host_requant_roster \
  --run-dir /tmp/mx-host-object-run \
  --fresh-dir /tmp/mx-host-object-fresh-run \
  --source-root "$MX_REPRODUCTION/spike" \
  --out-dir /tmp/mx-host-object-evidence
```

The `fresh-dir` is the same qualification command run from an independent
published checkout. These are functional-model results for the eight source
shapes and selected profiles. They do not qualify RTL timing or an FPGA image.
