<a id="generated-vpu-and-resident-requant-stage-in-nicolas-attention"></a>

# Generated VPU and resident requant stage in reference attention

The [source baseline](../nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/README.md)
runs the unmodified 32×64×64 `attn_vpu.c` on pinned Spike. This replay uses
the same generated data and source checker, but replaces its **seven VPU
calls and one `spad_requant` call** with `mx_issue(p_scales)`. The issuer is
lowered from `attention.vector_bound.mlir`, which is bound to the current
model2MLIR QK→scale→softmax→PV graph, the original source schedule, the
source data header, and the selected MX+VPU profile.

The generated issuer has eight ordered commands: seven funct-33 VPU commands
and one funct-34 resident E4M3 requant command. Its RV64 object has **zero
allocated data bytes**. The transformed driver contains no handwritten
`gemmini_vpu_*` or `gemmini_spad_requant` call. Pinned Spike matched the
unmodified source's score, probability, requantized probability, and PV
hashes. Two runs produced an identical log. The archived object, issuer C,
bound MLIR, binding manifest, physical commands, transformed driver, log,
and receipt make the check reviewable.

The original source C still issues **QK and PV matrix commands** and moves
their operands. This evidence qualifies the compiler's VPU/requant middle,
not a fully compiler-generated attention program. The next gate is one typed
graph that also lowers both matrix stages and their scratchpad routing, then
matches these four stage hashes from a generated object on Spike.

The public object compiler reproduces the archived issuer. From the repo root,
set `RTL_ROOT` to the `gemmini-mx-cleanup` checkout at `266c593`, `RISCV_ROOT`
to the RV64 toolchain, and `OUT_DIR` to a fresh directory, then run:

```sh
python -m tools.compile_object \
  --mlir docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb/attention.vector_bound.mlir \
  --binding-json docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb/vector_binding.json \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --rtl-root "$RTL_ROOT" --riscv-root "$RISCV_ROOT" --out-dir "$OUT_DIR"
```

The generated `mx_issue.c` SHA-256 is `547abea3ce2019b57baa0c23897dd5747bed22cc0b1e7ec12cd462a5e1cfaccc`;
the object SHA-256 with the pinned toolchain is
`8750f21c1ee263afa2794d50a5b311d23b4338f0aea617677e656a8b5975314e`.
The compiler checks the eight typed operations and their source/frontend
binding before issuing commands. The object takes one runtime pointer,
`p_scales`; the source driver still provides that buffer and the matrix stages.

Run the archive check with:

```sh
python -m pytest -q tests/test_nicolas_attention_vector.py
```
