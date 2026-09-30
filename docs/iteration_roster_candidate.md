# MX Phase 0 iteration capture roster (candidate)

This roster proposes independent, small models for deriving the MX Phase 0
operation corpus. It is **not admitted**. The selected RTL for the next review
is Gemmini `2029218197f771ce71416f859d975bea47b7aabc` with MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`, as declared in the
[candidate software spec](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml).
The existing TorchAO adapter remains guarded to the older selected RTL; its
capture result cannot qualify the newer revision.

## Iteration inputs

All four models and their synthetic FP32 inputs are defined in
[`examples/iteration_workloads.py`](../examples/iteration_workloads.py).
`make_case(name)` initializes parameters and generates inputs with CPU seed 0.
The materializer records its source hash, PyTorch and Python versions, the
contract and policy hashes, and SHA-256 hashes for each state file, input file,
and original `torch.export` file. These files are candidate capture inputs;
their generated hashes, rather than a mutable branch name, identify a run.

| Case | Exact input shapes | MX-sized contractions | Host seams |
| --- | --- | --- | --- |
| `linear_seam` | tokens `[32,64]` | two Linears; M=32, K=64, N=64/32 | LayerNorm, GELU |
| `decoder_block` | tokens `[1,32,64]` | four Linears, two batched QK/PV matmuls with 32-token, 32-wide heads | causal mask, softmax, LayerNorm, GELU, residual |
| `vision_patches` | image `[1,3,16,32]` | two Linears on 32 patches of width 64 | patch Conv2d, GELU, residual, mean reduction |
| `policy_fusion` | image/text/state tokens, each `[1,32,64]` | four Linears, two functional matmuls across a 64-token context | concatenation, softmax, LayerNorm, residual |

The candidate policy is [`examples/default-policy.yaml`](../examples/default-policy.yaml):
MXFP8 at every eligible contraction, with no overrides or resident output
chains. This is a structural bringup choice, not a reviewed model accuracy
policy. All contraction dimensions in this roster meet the candidate contract's
MXFP8, MXFP6, and MXFP4 tile bounds. FP6 requires reviewed site codebooks
before it can be selected. The host operations above need separate numerical
and transfer review.

Materialize one candidate input set under an explicit artifact root:

```sh
python -m examples.iteration_workloads \
  --output-root out/mx-iteration-candidate-20260929 \
  --contract mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml \
  --policy examples/default-policy.yaml
```

The output root must not exist. Retain `roster.json`, each `state.pt`,
`inputs.pt`, and `original.pt2` together. Review the generated hashes and
source closure before choosing an admitted capture. For later runs, use a new
artifact root rather than overwriting a previous set.

As a shape check only, the current adapter on its **older** selected contract
captured all 16 candidate contraction sites with the FP8 policy: 2 in
`linear_seam`, 6 in `decoder_block`, 2 in `vision_patches`, and 6 in
`policy_fusion`. None was skipped. This does not establish numerical accuracy,
latest-RTL compatibility, executable MX lowering, or a Phase 0 corpus.

## Held-out claim boundary

The independent iteration models are not copies or extracts of the held-out
full models. Keep the existing Merlin claim roster (`resnet50`, `tiny_llama`,
`smolvla`) out of corpus derivation. A full Pi0 capture is an additional
**proposed** held-out validation model; it is not in the current Merlin claim
roster. Select its checkpoint, input dataset, preprocessing, and ownership
before adding it. Do not use held-out model traces or op frequencies to tune
the iteration corpus.

[`pi0-quant`](https://github.com/chloe-wong/pi0-quant/tree/8bc0f6a6cb3faa237eb20b2354a0c8bc7afa524e)
and [`smolVLA-quant`](https://github.com/chloe-wong/smolVLA-quant/tree/337363277c347927c45121a40c8ed049d72cff82)
inform the coverage and later whole-model accuracy questions. Their current
per-tensor power-of-two FP8 experiments do not define this RTL's per-32 MX
quantization. [`microscaling-quant`](https://github.com/chloe-wong/microscaling-quant/tree/ea3f5bb4ee274e747ebdc70f463314b104bb45d0)
is the independent MX operand reference used in the
[numerical review](phase0_review.md); it is not an iteration model capture.

## Admission gates

To promote this roster, review exact artifact hashes and source closure,
select a model-level site/host policy, retarget the TorchAO kernel to the
source-bound latest RTL, run model2MLIR capture and check every contraction
site, derive an MX-only conformance profile, and run the required L0–L3
oracles. Full-model accuracy and executable compiler claims require their own
evidence after Phase 0. The [Phase 0 review](phase0_review.md) tracks the
remaining latest-RTL simulator and protocol work.
