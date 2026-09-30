# MX Phase 0 iteration capture roster (candidate)

This roster proposes independent, small models for deriving the MX Phase 0
operation corpus. It is **not admitted**. The selected RTL for the next review
is Gemmini `2029218197f771ce71416f859d975bea47b7aabc` with MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`, as declared in the
[candidate software spec](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml).
The OOT TorchAO operand adapter now accepts this exact revision pair for
explicit candidate capture. Its results do not qualify accelerator execution.

## Iteration inputs

All four models and their synthetic FP32 inputs are defined in
[`examples/iteration_workloads.py`](../examples/iteration_workloads.py).
`make_case(name)` initializes parameters and generates inputs with CPU seed 0.
The materializer checks the selected RTL source record against the pinned
Gemmini and MxGen checkouts, then records that source check, its own source
hash, PyTorch and Python versions, the contract hash, and SHA-256 hashes for
each state file, input file, original `torch.export` file, and derived site
inventory. It does **not** choose a precision policy. These files are candidate capture inputs;
their generated hashes, rather than a mutable branch name, identify a run.

| Case | Exact input shapes | MX-sized contractions | Host seams |
| --- | --- | --- | --- |
| `linear_seam` | tokens `[32,64]` | two Linears; M=32, K=64, N=64/32 | LayerNorm, GELU |
| `decoder_block` | tokens `[1,32,64]` | four Linears, two batched QK/PV matmuls with 32-token, 32-wide heads | causal mask, softmax, LayerNorm, GELU, residual |
| `vision_patches` | image `[1,3,16,32]` | two Linears on 32 patches of width 64 | patch Conv2d, GELU, residual, mean reduction |
| `policy_fusion` | image/text/state tokens, each `[1,32,64]` | four Linears, two functional matmuls across a 64-token context | concatenation, softmax, LayerNorm, residual |

The historical v1 candidate used [`examples/default-policy.yaml`](../examples/default-policy.yaml):
MXFP8 at every eligible contraction. It was a structural bringup choice, not a reviewed model accuracy
policy. All contraction dimensions in this roster meet the candidate contract's
MXFP8, MXFP6, and MXFP4 tile bounds. The original export has contiguous
operands at every selected contraction. FP6 requires reviewed site codebooks
before it can be selected. The host operations above need separate numerical
and transfer review.

Materialize one candidate input set under an explicit artifact root:

```sh
python -m examples.iteration_workloads \
  --output-root out/mx-iteration-layout-v2 \
  --contract mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml \
  --rtl-root /path/to/pinned/gemmini \
  --sources mx_gemmini_support/contracts/rtl_candidate_2029218.yaml
```

The output root must not exist. Retain `roster.json` and each case's
`state.pt`, `inputs.pt`, `original.pt2`, and `site-inventory.json` together.
The generated `policy_templates/` assign `host` explicitly to **every** site.
They are starting files for the operator, not reviewed numerical decisions.
Copy them into a separate policy directory and change exact site choices only
after reviewing their eligible formats, refusals, accuracy, and any FP6 codebooks.
Every policy must keep the inventory's `source_graph_sha256`. The selection
command checks every choice, rederives every inventory from frozen inputs,
and writes one digest-bound selection:

```sh
python -m examples.select_iteration_workloads \
  --roster-root out/mx-iteration-layout-v2 \
  --contract mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml \
  --rtl-root /path/to/pinned/gemmini \
  --sources mx_gemmini_support/contracts/rtl_candidate_2029218.yaml \
  --policy-dir out/mx-iteration-policies \
  --output out/mx-iteration-selection.json
```

The selector requires a host default and an explicit assignment for every
contraction, including host sites. It rejects unknown sites, changed graph
hashes, ineligible formats, and FP6 without reviewed codebooks. This freezes
an **operator choice**; it does not label the choice numerically correct.
Use a new artifact root for another candidate instead of overwriting these files.

Replay those exact files through the out-of-tree adapter into a separate
artifact root:

```sh
python -m examples.replay_iteration_workloads \
  --roster-root out/mx-iteration-layout-v2 \
  --output-root out/mx-iteration-capture-v2 \
  --contract mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml \
  --rtl-root /path/to/pinned/gemmini \
  --sources mx_gemmini_support/contracts/rtl_candidate_2029218.yaml \
  --selection out/mx-iteration-selection.json \
  --policy-dir out/mx-iteration-policies \
  --mx-opt out/build/mlir/tools/mx-gemmini-opt
```

The replayer checks every frozen file digest, saved export output, RTL source
record, contract, selected per-case policies, and a clean model2MLIR source checkout. It records the
model2MLIR commit, complete frontend trace, zero opaque operations, OOT
handoff validation and dialect verification, and hashes of the generated MLIR
and manifests. The
[historical v1 frozen roster receipt](evidence/iteration_roster_2029218.json) and
[capture receipt](evidence/iteration_capture_2029218.json) record one run with
model2MLIR `03718cb` and the `2029218` candidate. This earlier v1 run can
still be replayed with `--policy examples/default-policy.yaml`. All 16 FP8 sites were
selected: 2 in `linear_seam`, 6 in `decoder_block`, 2 in `vision_patches`, and
6 in `policy_fusion`; none was skipped. Each trace was complete with zero
opaque operations. The decoder's boolean mask uses model2MLIR's
target-neutral `logical_and` lowering. These are structural results on
synthetic inputs, not model accuracy, executable MX lowering, or a Phase 0
corpus.

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
select a model-level site/host policy, derive an MX-only conformance profile,
and run the required L0–L3 oracles. Full-model accuracy and executable
compiler claims require their own evidence after Phase 0. The
[Phase 0 review](phase0_review.md) tracks the remaining latest-RTL simulator
and protocol work.
