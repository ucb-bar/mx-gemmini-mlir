# A connected TinyLlama MLP section

This experiment uses real checkpoint activations from layer 5 of
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. It reruns one captured decoder layer,
checks the layer against its saved golden, and captures the pointwise section
`(gate * sigmoid_gate) * up` through model2MLIR main `905d2c9`. The sigmoid and
two projections are section inputs; they are not timed here. The section has
shape `[1,64,5632]`, or 360,448 elements.

Merlin outlines two typed multiply functions and selects their private
intermediate through its generic connected-dispatch planner. The target adapter
verifies the arithmetic and indexing. It requires the explicit
`bf16_rne_each_vpu_call` policy already used by the two original VPU kernels.
It keeps the BF16 intermediate in scratchpad, removing its host widening,
renarrowing, store and reload. All intermediate BF16 arithmetic rounding stays.
A range guard retains the original kernels for nonfinite or extreme inputs.

<a id="results-on-nicolass-spike-model"></a>

## Results on the pinned Spike model

| Case | Materialized warm cycles | Connected warm cycles | Result |
| --- | ---: | ---: | --- |
| TinyLlama MLP section | 22,759,408–22,760,183 | 16,878,060 | 25.8% fewer cycles |
| Independent 137-element tail | 9,605 | 6,822 | Every output bit matches |
| Extreme-input fallback, 137 elements | 9,605 | 15,350 | Original path runs; cost calibration rejects the optimization |

Each executable checks every output in one cold and two warm runs. Both
variants use the same input payload, checker, and timer objects. The TinyLlama
case removes 1,441,792 bytes of intermediate device DMA per invocation; measured
load bytes decrease by 720,896 per invocation and store puts halve. Host
packing, the range guard, and final widening are inside the timed function.
Native `mx-gemmini-opt` also accepts the emitted typed VPU modules.

This compares compiler output before and after optimization, not the reference implementation’s
stitched kernels. The clock is the reference implementation’s `e4m3_vpu` Spike timing model with
`GEMMINI_MODE=both` and `mem.host_tracking=0`; it is not FPGA timing or a full
Rocket memory-latency measurement.

## Numerical and integration limits

Every output is bit exact to the original two-call BF16 policy, not to strict
FP32 PyTorch multiplication. That policy has about 0.3685% relative L2 error
against FP32 on these section inputs, before and after the optimization.
This change does not repair full-model quantization quality.

The section uses checkpoint-backed FP32 layer activations, not a captured trace
from the current quantized full-model executable. In the original full graph,
an independent up-projection appears between the two multiplies. Wiring this
into whole-model compilation needs a legal scheduling/grouping decision and
cost measurement under that execution's activations. The stage builder does
not yet enable this adapter. No whole-model saving is inferred.

Matrix→VPU→requantization chains need separate layout and numerical proofs.
Existing target-owned tilewise epilogue and resident-pair lowerers provide
their command implementations; they still need adapters to this shared pass.
Regular Gemmini's generated R48 backend has integer readout and its own
intra-region and cross-region opportunities, described in
[the generated backend](https://github.com/ucb-bar/gemmini-mlir/blob/feat/phase1-el4-r48-compiler/phase1_generated/README.md).

## Verify and reproduce

The archive includes compressed input/golden payloads and generated C, typed
MLIR, plans, logs and source hashes. It contains no checkpoint weights or ELFs.
The portable verifier checks every golden element with an independent scalar
BF16 oracle and checks output hashes, receipts, traffic and timing selection:

```sh
python -B docs/evidence/connected_model_section_20261016/verify.py
```

To capture another section from a compatible checkpoint-backed layer bundle:

```sh
python -m tools.prepare_connected_tiny_mlp \
  --layer-bundle "$LAYER_BUNDLE" --model2mlir-root "$MODEL2MLIR_ROOT" \
  --out-dir "$CAPTURE_DIR"
python -m tools.qualify_connected_elementwise_spike \
  --capture "$CAPTURE_DIR" --boundary-policy bf16_rne_each_vpu_call \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$NICOLAS_MX_RTL_ROOT" --riscv-root "$RISCV_ROOT" \
  --extlib "$NICOLAS_GEMMINI_EXTLIB" --startup-dir "$BENCHMARK_COMMON" \
  --out-dir "$QUALIFICATION_DIR"
```

Put fresh output directories inside the task temporary directory. Capture used
two Torch threads and about 1.03 GiB peak RAM; the benchmark embeds only section
tensors and needs no full-model executable.
