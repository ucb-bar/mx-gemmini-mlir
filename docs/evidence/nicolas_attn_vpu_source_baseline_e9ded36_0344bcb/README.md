# Nicolas dense MX+VPU attention source baseline

This archive pins `npu-exploration` `0344bcb`'s unmodified `attn_vpu.c`
and its deterministic default data: `Sq=32`, `Sk=64`, `d=64`, seed `0`.
The source executes QK on MX, scales scores and applies softmax with seven
VPU commands, requantizes the probabilities into a resident E4M3 operand,
then executes PV on MX. The source data header is compressed here; its
uncompressed SHA-256 and the original generator hashes are in `receipt.json`.

The pinned Spike extension ran the original C source twice. Both runs had an
identical log. Its QK output matched every one of the source mesh-golden
words, and the score, probability, requantized probability, and PV hashes
matched the recorded expectation header. The source printed `attn_vpu PASSED`.
The archive contains the log and header, but not the source ELF. Its digest
and the selected RTL, Spike, profile, and source revisions are recorded.

The current `model2MLIR` main revision `e9ded36` also captured a PyTorch
`bmm(Q, Kᵀ) → softmax(score / 8) → bmm(P, V)` graph at the source shapes.
The capture has no opaque operation. `attention.model2mlir.mlir` and the
checked contraction worklist identify two 32×64×64 batched matmuls. This
is the frontend graph for the next compiler-generated connected object.

**Scope:** this is a source oracle and a portable frontend capture. The
archive does not contain a compiler-generated connected `attn_vpu` object or
prove parity for one. The existing connected object lowerer supports square
64/128 matrix pairs and scalar VPU steps; this source requires a 32×64×64
rectangular pair, seven reduction/broadcast steps, and BF16 PV readout.

The source generator's hard-coded output and sibling paths were redirected
in copies under the task's temporary directory, leaving its quantization and
mesh-model functions intact. The original `attn_vpu.c` was compiled without
edits for the hash-checked runs. No FPGA execution was used.

Check the archived graph, generated header, and source receipt with:

```sh
python -m pytest -q tests/test_nicolas_attn_vpu_source_baseline.py
```
