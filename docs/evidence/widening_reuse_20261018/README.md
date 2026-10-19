# Share packing through identical BF16 widening operations

The generic Merlin preparation pass now recognizes separate pure elementwise
extensions of the same SSA input through contiguous shape views. Static types,
identity indexing and the complete scalar body must match; the output initializer
cannot be read. Narrowing, different source tensors, transposes, extra arithmetic
and effect barriers retain their original execution. Target packing and consumer
arithmetic remain MX-owned. Selection remains the explicit experimental
`--share-activation-packing` option.

Four admitted, uninstrumented projection-stage pairs preserve every output word:

| Rows × input width | Format | Original cycles | Shared cycles | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 × 37 | FP8 | 205,138 | 74,505 | 63.6805% |
| 1 × 37 | FP4 | 401,665 | 139,674 | 65.2262% |
| 19 × 37 | FP8 | 527,914 | 232,868 | 55.8890% |
| 19 × 37 | FP4 | 535,277 | 234,655 | 56.1619% |

Each capture uses three BF16 linear projections with output widths 23, 41 and
17, a shared narrowed input and a concatenated F32 output. Both matrix tails
and one-token padding are exercised. Each candidate shares one preparation
among three consumers and removes two preparations. The timer includes host
conversions, allocation and packing. These are complete projection stages;
they establish no whole-model speedup or independent source-quality result.

The archive retains both captures, all eight actual ELFs, complete native logs,
original build/admission/compilation receipts, policies, generated C/IR, byte
preparations and relevant source snapshots. The manifest pins every member and
compiler revisions. It does not claim complete compiler dependency closure or
independent instruction-policy/hardware qualification. The loaded extension is
the unchanged pinned library in the earlier
`nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical` archive.

Run `python3 docs/evidence/widening_reuse_20261018/verify.py` to reopen all retained
members, actual ELF/admission hashes, complete output words/digests and cycles.
It uses only the Python standard library and does not extract temporary files.

Reproduce captures with `tools.capture_shared_projection_probe --rows 1` or
`--rows 19`, `--width 37 --bf16-projections`, the pinned profile and a fresh
`--out-dir`. The default generated policy selects FP8; the separate retained
`policy_fp4.json` selects FP4 for all three matrix sites. Build each matched pair
with `tools.build_merlin_stage`, identical capture/policy/profile and simulator
settings, checkpoint prepacking, unused-parameter pruning, structural ELF
admission and complete output caps of 81 or 1539. Add
`--share-activation-packing` only to the candidate. Verify with
`tools.compare_shared_preparation`, using the baseline first.

The rule also finds 24 groups/32 removed preparations in the compiled SmolVLA
flow stage and 32 groups/48 removed preparations in its prefix stage. Those
structural matches are not inferred performance or numerical results; the full
1+10+1 session measurements remain separate.
