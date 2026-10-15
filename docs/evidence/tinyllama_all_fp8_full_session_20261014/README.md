# TinyLlama complete all-FP8 MX+VPU Spike session

This archive records one **128-token prefill followed by 32 cached decode
calls** from the model2MLIR TinyLlama capture. Both stage catalogs bind all
200 matrix sites to direct E4M3 MX operands: 155 regular sites are selected
by the archived override maps, and 45 attention contractions use the
capture's FP8 choice. Eligible softmax, add, elementwise, unary, and
reduction regions also lower to the VPU. The selected hardware profile is
[`MxE4M3Fp4VpuGemminiRocketConfig`](../../../profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json).

On Nicolas's pinned MX+VPU Spike, all 33 steps completed with
**72,901,458,175 modeled cycles** under `e4m3_vpu` and
`mem.host_tracking=0`. All 99 output digests and all 33 full logit dumps are
present in the [run receipt](run.json). The
[FP32 comparison](against_fp32.json) checks the captured fixed synthetic
inputs: prefill logit cosine is **0.774153** and its top token differs from
FP32; mean decode cosine is **0.662794**, with **2/32** top-token matches.
These numbers measure numerical drift on this capture. They do not establish
generation quality or a useful quantization policy.

The earlier mixed FP4/FP8 full session has different outputs and a different
precision policy. Its cycle count is not an equal-output speedup denominator
for this run. Nicolas's independent 160-token source program is a single
causal forward with a different host/VPU placement and timing boundary; its
source-only result is archived in
[`nicolas_tinyllama_onepass160_same_input_20261014`](../nicolas_tinyllama_onepass160_same_input_20261014/README.md).
The matched compiler one-pass run is still pending, so no compiler/source
cycle ratio is claimed here.

The archive includes the original [link](link.json), prefill and decode
build receipts, catalogs, precision overrides, the compressed
[Spike log](spike.log.gz), 33 output tensors, and both captured FP32 logit
references. `archive_manifest.json` records a SHA-256 for each copied file.
Run `python -B verify.py` with NumPy to check the receipts, catalog precision,
output bytes, numerical values, and cycle total without downloading the
checkpoint. The complete compilation and execution procedure is in the
[model guide](../../compile_e2e_models.md).
