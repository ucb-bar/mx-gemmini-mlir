# Generated VPU and resident requant stage in Nicolas attention

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

Run the archive check with:

```sh
python -m pytest -q tests/test_nicolas_attention_vector.py
```
