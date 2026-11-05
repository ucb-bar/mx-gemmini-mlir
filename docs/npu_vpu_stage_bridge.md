# Source-bound npu VPU rewrite probe

The npu-exploration `model_run.v2` recipe can choose a default VPU/host route
and override exact source regions. `tools.rewrite_npu_model_vpu_sites` takes
that recipe's preflight plan, the original recipe, the captured MLIR, and the
same MX target profile. MX reopens the source through Merlin and independently
validates the complete matrix and VPU site census before any rewrite.

The command passes the admitted VPU regions to the existing
`MXVPUPostOffload` transforms. It checks that each selected softmax, add,
elementwise, unary, and reduction site appears exactly once in its rewrite
receipt. The output directory contains the copied source, rewritten MLIR,
per-class rewrite receipts, and a bridge receipt with hashes and selected
regions. Host-selected sites and their reasons remain visible.

From the managed task containing both repositories, an example is:

```sh
PYTHONPATH=../merlin/src:. python3 -m tools.rewrite_npu_model_vpu_sites \
  --plan ../.tmp/small-llama-fp8-vpu-plan/model_quantization_plan.json \
  --run ../npu-exploration/config/run/model_fp8_e4m3_vpu.json \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --mlir docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593/small_llama.model2mlir.mlir.gz \
  --out-dir ../.tmp/small-llama-vpu-rewrite
```

The pinned small Llama capture selects and rewrites 83 sites: two softmax,
13 add, 43 elementwise, 20 unary, and five reduction sites. The bridge
receipt status is `source_bound_vpu_rewrite_only_no_model_elf_or_spike_parity`.
The probe uses a synthetic sidecar solely because these source rewrites hash
it; it does not compile a VPU catalog or an ELF. Matrix quantization choices
from the npu recipe are not applied by this probe. A complete stage must
connect those choices, the live activation provider, and the actual capture
sidecar before execution or performance claims. The same region map is the
`enabled_vpu_regions` input already consumed by the MX model stage builder's
catalog and `MXVPUPostOffload` paths.
