# MX Gemmini derivation and phase ownership

Fixed RTL, MxGen, contract, model, inputs, policy, and tool versions must give
the same Phase 0 artifacts. An agent may improve the tools or propose an
authored policy, but an agent is not in the Phase 0 derivation path. The RTL
remains authoritative for hardware facts. A software contract can declare
semantics and host policy that RTL extraction does not establish; its presence
alone is not evidence that those declarations are true.

## Authorship and selection

The `2029218` [candidate software spec](../mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml)
was first authored in this OOT repository in commit `b24adcb` on 2026-09-29,
then revised for source and protocol checks. It is not generated from Merlin's
`examples/mx_gemmini/` directory. That directory owns an experiment definition,
recipe, and descriptor; it has no copy of this candidate spec. The candidate
spec combines declared software semantics with fields checked against the
pinned Gemmini/MxGen source. It remains `unreviewed`. A successful
`compile_contract` projects its declarations; the separately invoked
`rtl_check` checks exact source hashes, format and scale fields, selected
rounding paths, candidate command fields, and the standalone config's literal
mesh, capacity, latency, LUT, and activation switches. The check resolves
the default config and its standalone overrides, then refuses contract claims
that contradict its mesh, scale block, LUT, or disabled built-in operations.
It does not derive all shape
bounds, host operation placement, transfers, or mesh numerical behavior.
Those are authored claims needing separate review and execution evidence.
Neither operation approves the spec.

The user or experiment operator explicitly selects the RTL/configuration,
candidate contract, model and representative inputs, and policy file for a
run. The responsible hardware and numerical reviewers establish which
software declarations and FP6 codebooks may be marked reviewed. Nothing
selects the newest branch or silently promotes this candidate over the older
active provider contract. Merlin's MX Phase 0 recipe explicitly selects this
candidate provider resource; its old in-tree snapshot is historical. The OOT
iteration workflow freezes the same contract and a separate per-model site
inventory. Merlin's Phase 0 derivation now checks the exact policy digest
against each materialized quantization manifest. This is byte selection,
not approval of the numerical policy or application corpus.

For the declared `mx_block_reference` engine, Merlin inventories the exact
`mlc/validate/mx_ref.py` file loaded by both numerical consumers. Frozen
replay preserves its path relative to the selected MLC root. The rest of that
checkout, including generated runs and third-party data, is outside this
numerical reference selection. The captured bytes establish source identity;
they do not certify agreement with the selected RTL.

The inventory is **computed**, not hand-picked: `derive_site_inventory`
exports that selected model with those selected inputs, identifies `nn.Linear`
modules and visible functional contractions, then evaluates every MX format
against observed rank, static shape, operand layout, batch compatibility, and
the selected contract's M/N/K bounds. Its exact site IDs, source-graph digest,
eligible formats, and refusal reasons are scoped to that export. Different
input shapes or graph specialization require a new inventory.
The v2 roster materializer records the selected RTL source-check result and
its source-record digest before exporting models. `select_iteration_workloads`
rederives the inventory from frozen state, inputs, and contract, then checks
the operator's exact per-site policy against it. Replay rechecks both inputs.

The **precision assignment is selected by the user/operator**, normally with
the numerical team's recommendation and reviewed FP6 codebooks. They provide
an explicit policy mapping site IDs to FP8, FP6, FP4, or `host`; a policy with
`default_format: host` is the selective starting point. Capture applies it
only where the inventory rules admit it, and refuses an ineligible explicit MX site.
The FP8-everywhere `examples/default-policy.yaml` is a structural bringup
input, not an approved whole-model precision choice. The policy's bytes and
the original graph digest are bound into the capture handoff.

| OOT owner | Current mechanism | Remaining boundary |
| --- | --- | --- |
| `contracts/` and `contract.py` | YAML input is compiled to one deterministic, digest-bound consumer view. | The software declarations are authored and remain `unreviewed`; compilation does not prove them. |
| `rtl_check.py`, `config_facts.py`, and source records | Selected source hashes, format/protocol fields, and effective standalone config scalars are checked against pinned RTL/MxGen bytes. Disabled nonlinear, normalization, and max-pool operations cannot be declared accelerator operations. | This is not complete elaboration closure or extraction of all runtime behavior. The config switches describe built-in Gemmini units, not every possible host or compiler implementation. |
| `build_receipt.py`, `structural_facts.py`, and Merlin's generic FIRRTL source selection | Present Scala/artifact bytes and commits are rechecked; exact FIRRTL is converted to SoC and Gemmini HW, and its hierarchy is cross-checked. The OOT adapter binds a dense 16×16 tile grid, per-tile multiplier hierarchy, bounded port-wiring witnesses, and selected built-in feature switches to the exact Merlin fact bundle as structural observations. | The build receipt was authored after elaboration, so historical Scala-to-FIRRTL input closure is still unproven. Port links do not prove complete arithmetic dependency and do not enter `facts.arrays`; the candidate is not an admitted Phase 0 fact bundle. |
| `capability_proposal.py` | Projects the selected grid and built-in switches separately from the software spec's per-format contraction and host intent into a digest-bound review artifact. | Its `candidate_not_admitted` output is not a Merlin target contract; the format, shape, and host rules remain authored and unreviewed. |
| `policy.py`, `legality.py`, `m2m_adapter.py` | Exact site choices and M/N/K bounds come from the selected policy and compiled contract. A source-bound inventory derives every site's eligible formats and refusal reasons; capture records selected, host, and skipped sites. | Format choices and FP6 codebooks are numerical policy inputs. Shape/layout eligibility is not proof of executable lowering. |
| `torchao_quant.py` | A reusable TorchAO handler and graph pass apply the chosen site format. The operand kernel is checked against selected format constants and narrow independent/RTL diagnostics. | Quantization arithmetic and graph rewrites are handwritten algorithms; mesh arithmetic and whole-model numerical agreement are not modeled here. |
| `layout.py`, `contraction.py`, `candidate_protocol.py` | Source-scoped packing and command fields are checked by narrow payload and simulator diagnostics. | General transfer scheduling, capacity planning, and executable model lowering remain open. |
| `handoff.py` and `lib/MxOps.cpp` | Digests, census, policy format, contract shape, FP6 codebook, and selected chains gate a verifier-readable MX operation plan. | The plan is not an executable replacement for the source MLIR. |

## Latest candidate derivation

Four synthetic iteration captures (`linear_seam`, `decoder_block`,
`vision_patches`, and `policy_fusion`) have complete source traces and
materialized capture receipts. Their exact, operator-selectable policy files
currently choose `host` for every contraction. This is a diagnostic selection;
no mixed-precision choice has been approved for these models.

The selected `2029218` fact bundle records a source-bound 16×16 mesh grid,
multiplier instance hierarchy, port-wiring witnesses for all 256 tiles, and
configuration switches: built-in nonlinear
activation, normalization, and max-pool are disabled, while the LUT is present
and enabled. `facts.arrays` remains empty. Merlin can
use the grid to size tile-edge probes without declaring a compute engine. A
fresh deterministic Phase 0 derivation inventoried 674 MLIR operations and
generated ten diagnostic entries: four saved model captures and six host or
composition probes. It has **zero admitted accelerator cells** because the
active OOT capability contract intentionally declares no compute unit. The
host capability screens are also `unknown`. The synthesis profile is a
candidate artifact, not a Phase 0 admission or a functional compiler result.

The OOT proposal can be reproduced from explicit inputs without an agent:

```sh
python -m mx_gemmini_support.capability_proposal \
  /artifact/facts-with-mesh.json \
  mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml \
  /artifact/capability-proposal.json
```

It records the RTL-derived grid and built-in switches separately from the
authored FP8, FP6, and FP4 ranks, layout and M/N/K bounds, accumulator dtype,
and host families. It
refuses a changed spec digest or mismatched FIRRTL identity. The next
capability step is to corroborate the connected arithmetic datapath and
review a compute-unit contract before selecting one for accelerator corpus
generation. Geometry, operand formats, shape bounds, host placement, and
numerical rules retain separate provenance; an instance hierarchy alone
cannot authorize contraction lowering. The
source-build receipt also does not prove historical Scala-to-FIRRTL input
closure. Numerical review, mixed-site policy selection, host semantics, and
L0–L3 execution remain open.

The aim is one generic implementation consuming derived facts and explicit
reviewed policies, with source checks that fail when a pin or declaration
changes. Generating Python and C++ files anew from RTL would still require
reviewed rules for numerical behavior, graph boundaries, host transfers, and
scheduling. Those rules cannot be inferred safely from a format name or a
passing single-tile test.

## Phase boundary

| Phase | MX responsibility | Required output |
| --- | --- | --- |
| 0 | Pin source and tools; extract and check hardware facts; review software-visible numerical and host semantics; freeze model/input and site-policy identities; derive candidate legality, typed transfer obligations, capsules, and independent goldens. | Reproducible contract, corpus, complete site census, and explicit unresolved obligations. No compiler success claim. |
| 1 | Implement and repair the OOT target lowering and runtime route. Decide legal placement, build Q/DQ and mixed-format seams, schedule scale/LUT/operand transfers and K waves, execute host fallbacks, and check emitted artifacts against Phase 0 capsules and target oracles. | A frozen reusable functional compiler with source-bound execution and numerical evidence for its admitted domain. An explicit MX override cannot be silently left on the host. |
| 2 | Start from that exact admitted Phase 1 compiler. Optimize tiling, fusion, resident requantization, transfer overlap, and other reusable rules; measure declared tuning and held-out model cohorts using paired, qualified engines and a separate performance policy. | Measured improvement and claim artifacts bound to the same functional baseline, hardware revisions, corpus, and toolchain. |

For mixed precision, Phase 0 derives the legal format set for each source
contraction. The numerical owner supplies the model-specific FP8/FP6/FP4
assignment and reviewed FP6 codebooks; the policy and original graph digest
become frozen inputs to a particular capture. Phase 1 must lower exactly
those selected sites and implement each intervening host/accelerator seam,
or refuse that candidate. It may explore another explicit policy as a new
candidate, with separate identity and correctness evidence. Phase 2 may
optimize an admitted policy's schedule. Changing a site's format or codebook
changes numerical semantics, so it requires renewed functional qualification
before its performance can be compared or claimed.

The current OOT package ends at a materialized Phase 0 capture/handoff diagnostic plus
bounded RTL bringup programs. It has no executable general MX lowering, host
runtime, or Phase 1 certificate. The default Merlin provider still selects the
older RTL pin; the `2029218` candidate is opt-in and `unreviewed`.
