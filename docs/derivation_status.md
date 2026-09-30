# MX Gemmini derivation and phase ownership

Fixed RTL, MxGen, contract, model, inputs, policy, and tool versions must give
the same Phase 0 artifacts. An agent may improve the tools or propose an
authored policy, but an agent is not in the Phase 0 derivation path. The RTL
remains authoritative for hardware facts. A software contract can declare
semantics and host policy that RTL extraction does not establish; its presence
alone is not evidence that those declarations are true.

| OOT owner | Current mechanism | Remaining boundary |
| --- | --- | --- |
| `contracts/` and `contract.py` | YAML input is compiled to one deterministic, digest-bound consumer view. | The software declarations are authored and remain `unreviewed`; compilation does not prove them. |
| `rtl_check.py` and source records | Selected source hashes and scoped fields are checked against pinned RTL/MxGen bytes. | This is not complete elaboration closure or extraction of all runtime behavior. |
| `policy.py`, `legality.py`, `m2m_adapter.py` | Exact site choices and M/N/K bounds come from the selected policy and compiled contract. A source-bound inventory derives every site's eligible formats and refusal reasons; capture records selected, host, and skipped sites. | Format choices and FP6 codebooks are numerical policy inputs. Shape/layout eligibility is not proof of executable lowering. |
| `torchao_quant.py` | A reusable TorchAO handler and graph pass apply the chosen site format. The operand kernel is checked against selected format constants and narrow independent/RTL diagnostics. | Quantization arithmetic and graph rewrites are handwritten algorithms; mesh arithmetic and whole-model numerical agreement are not modeled here. |
| `layout.py`, `contraction.py`, `candidate_protocol.py` | Source-scoped packing and command fields are checked by narrow payload and simulator diagnostics. | General transfer scheduling, capacity planning, and executable model lowering remain open. |
| `handoff.py` and `lib/MxOps.cpp` | Digests, census, policy format, contract shape, FP6 codebook, and selected chains gate a verifier-readable MX operation plan. | The plan is not an executable replacement for the source MLIR. |

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

The current OOT package ends at a Phase 0 capture/handoff diagnostic plus
bounded RTL bringup programs. It has no executable general MX lowering, host
runtime, or Phase 1 certificate. The default Merlin provider still selects the
older RTL pin; the `2029218` candidate is opt-in and `unreviewed`.
