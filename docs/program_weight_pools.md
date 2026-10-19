# Link programs with different remaining weight layouts

Independent stage pruning can leave different immutable parameter subsets in
each compiled stage. Select `--pack-program-weights` on
`tools.link_merlin_session` to use Merlin's explicit program-local weight pool.
The default linker continues to require identical nonempty weight blobs.

Merlin derives each aligned offset from the generated argument tables, emits
the corresponding scheduler calls, rederives every capture and selected entry
projection, and verifies every weight byte before assembling the pool. Complete
identical blobs share a span; different layouts retain separate spans. Existing
stage blobs can supply storage only through the normal complete-byte checks.
The link receipt retains the pool layout and the final blob's size and hash.

Use the same pool option for both measured candidates and retain
`--measure-preparation`, the normal final ELF admission and the original session
schedule. State bindings, output ordinals and numerical gates keep their
existing meaning. This layout is not a numerical or performance certificate.
