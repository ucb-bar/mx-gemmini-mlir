# Sharing immutable packed constants across session stages

`tools.link_merlin_session --share-catalog-constants` explicitly pools equal
packed checkpoint operands across stage catalogs. It is experimental and off by
default. The ordinary stage route remains available; selection refuses if the
required proofs fail or the complete padded pool does not reduce storage.

Merlin owns the representation-aware byte pool, aligned layout and per-stage
symbol bindings. The MX provider owns encoded layouts, the native DMA contract,
object splitting, ABI relocation and final placement. Selection follows actual
bytes, representations and uses, independently of model names or provenance
labels. Equal samples or matching tensor shapes do not authorize sharing.

The provider reconstructs each original catalog from all saved components and
requires complete object identity. Every selected reader is regenerated through
the owned native emitters and recompiled under owned ABI options and headers;
the resulting objects must match the original bytes. These templates use fixed
operand addresses solely for aligned, readonly, synchronous native DMA. All
references must close in those readers. Model objects, ABI shims and common
runtime objects cannot gain unqualified access. Existing symbol/address
observers, including undefined symbols, refuse.

Pure constant objects are reopened against every blob, symbol, alignment and
padding byte using fresh assembler metadata. Extra allocated storage, code,
relocations and exports refuse. Generic pooling compares every complete original
file, preserves the strongest alignment, keeps different representations
separate and rechecks inputs after assembly. The new pool object receives the
same complete-content check. Only the constant component leaves each catalog;
other readonly data and mutable workspaces remain.

The linker redirects qualified imports, places the pool within selected RISC-V
address reach, recomputes arena bounds and runs existing final ELF admission.
Captured inputs, output ABI and timing intervals retain their original scope.
A selected proof or admission failure stops that attempt.

Add the flag to the ordinary link, retaining all original options, especially
host-math guards and full output dumps:

```sh
python -m tools.link_merlin_session \
  --session-root "$CAPTURE" \
  --stage-build "prefill=$PREFILL_BUILD" --stage-build "decode=$DECODE_BUILD" \
  --workdir "$TMPDIR/fresh_pooled_link" \
  --profile "$PROFILE" --merlin-src "$MERLIN_SRC" --riscv-root "$RISCV_ROOT" \
  --structural-elf-admission --measure-preparation --share-catalog-constants
```

Use a fresh work directory and retain the selected `MERLIN_READELF` and compiler
paths. Component/reader proofs, blob bindings, actual pool identity, tool/producer
pins and commands are saved in `constant_pool/receipt.json` and the session
receipt. Saved compiler-option strings cannot authorize proof compilation; the
provider uses its owned options and requires exact original machine objects.

Small native qualifiers cover two partial-tile shapes, three changed inputs,
all output words and canaries, two admitted executables and complete work
counters. Refusal cases cover changed data, source/header changes, extra address
observers, namespace capture and absent savings. These are not whole-model
performance or independent hardware qualification. Complete cached-session
output, numerical and cost comparison remains required before broad enablement.
