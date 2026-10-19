# Qualified immutable constant sharing

The explicit cross-stage pool has passed native ownership, relocation and
execution qualifiers. Two independent admitted executables preserve every
output and guard across three changing inputs and two partial-tile matrix
shapes. This is a correctness/resource qualifier; it has no section timer and
does not establish a whole-model performance result.

| Native fixture resource | Original | Pooled |
|---|---:|---:|
| Encoded constant storage | 4,224 bytes | 2,112 bytes |
| Actual executable | 30,416 bytes | 28,192 bytes |
| Native loops | 6 | 6 |
| Mesh tiles / busy cycles | 48 / 768 | 48 / 768 |
| Load bytes | 18,432 | 18,432 |
| Scale bytes | 576 | 576 |

Both shapes use N=17 and K=35, with M=5 and M=16. The physical native partition
retains the original padding and reduction order. Three input values (1, −1,
0.5) independently imply full output values 35, −35 and 17.5. The reader reopens
all **1,071 F32 words per executable**, not just checksums or error summaries.
The native program also checks all prefix/suffix guards. Complete work counters
account for all six calls; no timing ratio is inferred.

The archived pool receipt binds the complete original catalogs, ordered
component identity, exact fresh source/object reader proofs, complete undefined
reference closure, generic representation/byte bindings, actual pure constant
objects, tool/producer pins and commands. The reader reopens the real readonly
ELF section and all exported constant offsets, compares all source bytes and
padding, and checks both actual executable byte streams and admission records.
Fresh replay receipts pin the selected Spike and unchanged extension
`5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5`.

Core regression coverage includes unknown/writable/address-observing contracts,
changed duplicates, symlink replacement, hash collisions, differing complete
values/representations, alignment and actual native host linking. Target cases
reject changed reader sources/headers, changed packed data, extra compiled or
model address observers, namespace capture and absent storage savings. Proof
compilation uses owned ABI options; saved command strings cannot interpose a
different header. The focused core suite passed 37 tests and target/session
suite passed 42 tests, including both admitted native executions. Structure,
docs, full target-name and staged regex checks passed. Existing unrelated full
regex debt was not changed or claimed to pass.

Verify with the Python standard library:

```sh
python docs/evidence/shared_catalog_constants_native_20261019/verify.py
```

Complete verification passed. The archive retains **103 files in 59,668
compressed bytes**, including generated C, source arrays, assembly, all small
objects/images, full native output streams, admissions, replay commands,
producer source snapshots and checks. Toolchains, startup sources and the large
original model artifacts remain external; this is not a hermetic rebuild or
independent hardware qualification.

The archived input inventory separately reopens all 310 actual code/scale pairs
for the latest full prefill/decode captures: 155 prepacked matrices and
1,066,745,856 duplicate payload bytes. A full-size feasibility proof reconstructs
both original catalog objects exactly and regenerates/recompiles 625 selected
objects per stage to exact original bytes. These observations establish input
applicability; the inventory is not a merged full-model image. The separately
launched full pooled cached link/execution remains pending. No new complete
session cost or image-size claim is made here.

For a fresh native execution, run `tests/test_catalog_constant_pool.py` with
explicit `MERLIN_TEST_RISCV_ROOT`, `MERLIN_READELF`, `MERLIN_TEST_STARTUP` and
`MERLIN_TEST_EXTLIB`, `PYTHONPATH=../merlin/src:../model2mlir:.`, and a fresh
task-local pytest `--basetemp`. The native execution test constructs both actual
images, preserves structural admission and runs the selected simulator. For
full-session reproduction retain the original guards, dump settings and timing
options and add `--share-catalog-constants` to the ordinary link, as described in
[the pooling contract](../../shared_catalog_constants.md). The default route
remains unchanged; broad enablement requires complete-model qualification.
