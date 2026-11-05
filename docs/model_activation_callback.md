# Live user quantization at a dense model site

The dense Merlin catalog entry receives a live, contiguous `float *lhs` from
the executing graph. `bind_model_activation_callback` checks the user C source
and every `host_c_v1` header against the npu-exploration run recipe. Passing
the result to `emit_dense_contraction(..., activation_callback=...)` generates
an adapter that calls `npu_mx_quantize_activation_v1` on that live pointer
before the MX issuer can load A. The callback receives the exact source
region, direct FP4 E2M1 or FP8 E4M3 format, logical/padded dimensions,
code/scale capacities, and canonical settings JSON. It owns the numerical
rule and complete padded A output. The adapter rejects a callback error or
an unwritten/invalid E8M0 scale byte. All direct code bytes are legal, so a
provider that falsely returns success after underfilling codes violates the
ABI but cannot be detected from the code bytes alone.

`bind_callbacks_from_model_plan` is the npu entry seam: it checks that the
selected `model_run.v2` plan is byte-identical to an MX structural admission
receipt, then selects exact direct rank-2 FP4/FP8 sites and binds the recipe's
guest source and headers. It rejects a changed plan or unsupported selected
mode. This binder consumes the user's policy; it does not choose precision or
quantization numerics.

For a full catalog build, pass a mapping of exact source regions to bound
callbacks as `activation_callbacks` to `make_merlin_catalog_builder`. This
first bridge requires rank-2 contractions with immutable prepacked B and
plain dense transport. The planner gives each selected region its own kernel
symbol even when other sites have the same shape. The builder copies the
verified ABI/custom headers, compiles the pinned C source, and records its
hash and the selected region in `mx_catalog.json`. The generated entry exports
`<kernel_symbol>_host_quant_cycles`, measured from the guest cycle counter.
This cost belongs in the full end-to-end model timing.

The host C test in `tests/test_model_activation_callback.py` calls the same
generated dense entry twice with a changed live `lhs` value and observes
different A codes at the issuer; it also checks source hash refusal. A real
small Llama captured-model planning test proves the selected region gets a
separate kernel. These tests do not build an RV64 catalog or execute Spike.
The matching toolchain and extension are required before guest object and
numerical claims. The new npu model recipe is not yet wired automatically
into the Merlin model CLI; a caller must pass the validated site mapping.
