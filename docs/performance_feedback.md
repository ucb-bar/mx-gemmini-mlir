# Performance feedback and hardware identity

The published model results use Rocket-Spike with a pinned Gemmini extension.
They do not come from an FPGA bitstream. FireSim workload packaging has been
checked, but no deployed bitstream identifier or FPGA measurement is qualified.

## Locate the bottleneck

Start with complete sessions, then break down the expensive stages and layers.
Keep the checkpoint, input stream, numerical policy and target fixed while
comparing implementations.

| Feedback | What it can reveal | Current support |
| --- | --- | --- |
| Reset, prefill, every decode, and state-handoff cycles and instructions | Expensive setup, copying, and growth with context length | Session linker `--measure-preparation` and complete run receipts |
| Host/MX/VPU time by source operation and layer | Packing, normalization, softmax, layout conversion, or host fallbacks dominating execution | Separate stage builds with `--op-profile`, summarized by `tools.report_stage_profile` |
| Matrix tiles, busy cycles, shape and padding | Wasted work for narrow matrices, small batches, or attention padding | Extension counters and generated catalog shapes |
| VPU commands and logical elements per call | Short vector launches and repeated conversions | Extension counters, VPU catalogs and generated call sites |
| Load bytes, scale bytes, Gets/Puts and mean load latency | Repeated operand preparation, scale overhead, or transfer pressure | Extension performance report |
| L2 hits/misses, bus/DRAM busy cycles, scratchpad and accumulator port activity | Memory-system and local-port pressure | Extension performance report |
| Fences and host-stall cycles | Frequent synchronization or an issuer waiting on the device | Extension performance report |
| Queue occupancy, stall reasons, bank conflicts and a CPU/DMA/MX/VPU timeline | Why work fails to overlap and which resource blocks progress | Additional cycle-level simulator or RTL instrumentation is needed |

`tools.report_stage_profile` currently parses a subset of the aggregate counters.
Retain the complete `perf.txt` as well as its summary. The existing source-region
mapping associates profiled matrix calls with captured operations; layer names
and operation identities are available in the capture and catalog.

An instrumented build is a separate measurement. Some fusion and grouping
options explicitly refuse `--op-profile`; disabling them changes the executed
implementation. Record those differences, compare the complete outputs, and
use the uninstrumented session for its end-to-end result. A profile of a
different schedule cannot establish the optimized schedule's exact breakdown.

Busy-cycle counts are useful work indicators, but their ratio to elapsed cycles
is not automatically physical PE utilization. Engine intervals may overlap.
Do not add overlapping CPU, DMA and accelerator times to explain an elapsed
total. A common timestamped timeline is needed to identify the critical path.

Decode deserves separate attention: a single-token projection has different
reuse and padding from a 128-token prefill. Report useful versus physical matrix
work, bytes per generated token, cycles per decode position, and cache-update
cost. These distinguish insufficient matrix occupancy from memory bandwidth
or host preparation limits.

The published timing preset is `e4m3_vpu`, with `GEMMINI_MODE=both` and
`mem.host_tracking=0`. Host memory traffic is not tracked in those measurements.
That limits conclusions about CPU/device memory contention. Separately recorded
experiments with supported host-memory tracking and varied memory latency or
bandwidth can test sensitivity; they are different timing configurations.
Hardware counter measurements must ultimately validate the simulator's trends.

## Inspect the effective simulator settings

The [retained configuration dump](evidence/performance_configuration_20261023/config_dump.txt)
comes from the actual pinned extension, with the same preset and override used
for model measurements. The [receipt](evidence/performance_configuration_20261023/receipt.json)
records binary hashes and scope. The probe runs CPU code only; its zero-command
performance report is not an accelerator validation result.

Set `GEMMINI_PERF_DUMP_CONFIG=1` on an otherwise unchanged Spike invocation to
print the effective settings. Retain the dump with the command, binary hashes,
environment and run receipt. A preset name alone is insufficient.

| Setting | Effective value | Interpretation |
| --- | --- | --- |
| Mesh; VPU units | 16×16; 2 | Agrees with the compiler profile's basic geometry |
| Load/execute/store controller queues | 8 / 8 / 2 | Agrees with the compiler profile; reservation-station queues are separate |
| Scratchpad | 4 banks × 4096 rows × 16 bytes | 256 KiB in the timing model |
| System bus; modeled DRAM throughput | 64 bytes; 8 bytes/cycle | Bus width and sustained DRAM throughput are different quantities |
| L2 capacity; hit latency | 512 KiB; 12 cycles | Timing-model settings; complete cache topology still needs hardware verification |
| DRAM latency; model | 20 cycles; fixed pipeline | Does not establish equivalence to the Chipyard outer-latency setting or a deployed FireSim memory model |
| Host memory tracking | Disabled | CPU/device contention is not fully represented |

The same binary's separate `firesim` preset has different defaults:
[the retained probe](evidence/performance_configuration_20261023/firesim_preset/receipt.json)
reports a 32-byte bus, 30-cycle DRAM latency and at most 16 outstanding DRAM
reads. The selected `e4m3_vpu` configuration instead has a 64-byte bus,
20-cycle latency and no read cap. The compiler profile specifies a 64-byte
bus. Neither preset's name establishes the settings of a deployed image;
match the actual hardware and runtime memory configuration before selecting
or overriding a timing preset.

The dump documents several assumptions and approximations alongside calibrated
values. Accumulator row packing, cache topology, outer-memory timing and all
version-dependent behavior still require a source/build-manifest match. Basic
geometry agreement does not qualify the complete timing model against RTL or
a bitstream. Record any calibration or sensitivity experiment separately from
the published configuration.

## Match the target before using a bitstream

The intended configuration is
[`MxE4M3Fp4VpuGemminiRocketConfig`](../profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json):

- 16×16 mesh; FP8 E4M3 and FP4 E2M1 matrix modes; 32-element scale groups.
- 256 KiB scratchpad, 64 KiB accumulator, 16 KiB scale memory, and VPU enabled.
- One Rocket core, 512-bit DMA/system bus, four L2 banks, and all remaining
  resource and Chipyard overrides recorded in the profile.
- Gemmini revision `266c593f2cb51d7e3fe83fc0317072b585ac3c52` and mxgen revision
  `dba3e7e706efa156d96efe595cd9935c825ea14d`, with individual RTL source hashes.

The profile retains `qualification: structural_unqualified`. Matching a name,
JSON file, or opcode smoke test does not qualify an FPGA implementation.

For a source checkout at those revisions, verify the complete profile against
the actual RTL files, including the Chipyard wrapper and nested mxgen source:

```sh
export GEMMINI_RTL_ROOT=/path/to/pinned/gemmini
python - <<'PY'
import os
from pathlib import Path
from mx_gemmini_support.target_profile import load_profile, profile_sha256

profile = load_profile(
    Path("profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"),
    rtl_root=Path(os.environ["GEMMINI_RTL_ROOT"]),
)
print(profile_sha256(profile))
PY
```

This check rejects a different checkout revision, source bytes or capability
set. It does not validate a previously built bitstream. To do that, retain the
FireSim hardware DB entry, build recipe and source manifest, generated hardware
parameters, bitstream/AGFI identifier, runtime configuration, and loaded image
identity. Compare the mesh, formats, VPU, memories, bus, core and DRAM address
range against the compiler profile and packaged executable's requirements.
Run small format/VPU/output checks before the complete model, then retain all
output/state and timing observations from the actual deployment.

The [compilation guide](compile_e2e_models.md#package-a-session-for-firesim)
describes packaging and the remaining platform console/exit checks. Publish
FPGA cycles with their own hardware identity; keep them separate from the
existing Spike measurements.
