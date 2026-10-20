# Recurrent source replay and full-run artifact checks

Two independent recurrent smoke tests pass exactly, using two and eight query
heads per K/V head. Each two-layer Llama model executes 64 prefill tokens and
33 single-token decodes. All 34 complete logit
tensors and the final unquantized K/V cache match a separate NumPy replay
using the pinned Radiance mesh and Rocket VPU references. The archive retains
the target executables, inputs, port sources, replay sources, all 578 mesh
reference invocations per case, complete output bytes and receipts.

A separate graph-level reference also passes for the eight-to-one GQA case.
It reads the original prefill's parameter bindings and executes Llama operations
on logical tensors. It does not interpret the generated decoder descriptors.
Every logit and final K/V word agrees, checking the scheduler's head assembly
and cache updates in addition to the port's kernel arithmetic.

It also retains fresh audits of the existing full-model results: the winning
160-token TinyLlama forward's actual payloads, cached-session executable
recovery to the original hashes, full-session artifact verification, and
independent recurrent-state checks for cached TinyLlama and both SmolVLA
compiler versions. The recovered executables were not remeasured by this
audit. Missing files and the rejected incomplete-catalog recovery attempt are
recorded in `checks/cached_recovery.json` within the archive. Duplicate inactive
link objects use exact prefix/suffix fragments and a hash-pinned retained
checkpoint object; their original complete hashes were checked before storage
compaction. The running simulator's executable was not changed.

The pending full cached port's encoded weights also pass a preflight check:
all 155 learned matrices agree with both compiler stages in the actual linked
compiler executable. This checks every FP8 code and E8M0 scale after undoing
the source layout. It does not establish completed full-model execution.

```sh
python docs/evidence/independent_session_checks_20261020/verify.py
```

The verifier needs Python and NumPy. It verifies every archived member and
compares the complete observed and independently replayed smoke outputs.
Identical files are stored once; `artifacts.json` maps their logical names.
This archive does not establish a new full-session speedup. See the
[session-check guide](../../independent_session_checks.md) for the full cached
run, its predeclared numerical criteria, and the remaining SmolVLA baseline.
