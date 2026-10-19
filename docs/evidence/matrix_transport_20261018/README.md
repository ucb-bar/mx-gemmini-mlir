# Private matrix-to-VPU transport

The compiler can preserve BF16 matrix readout through private shape-only views
into a VPU add. Public outputs remain FP32. It removes intermediate widening and
renarrowing; it does not keep values resident in device scratchpad, fuse device
commands or remove fences.

| Complete section | Ordinary boundary | BF16 transport | Output words |
| --- | ---: | ---: | ---: |
| Real TinyLlama first-layer down-projection + residual, 160 rows | 85,048,339 | 81,116,526 | 327,680 |
| Same section, one row of the source activation | 7,368,960 | 7,354,638 | 2,048 |
| Independent 19×37 projection to width23 + residual | 161,860 | 158,832 | 437 |
| Independent one-row projection + width23 broadcast bias | 68,159 | 68,023 | 23 |
| Two projections sharing one packed activation, then two residual adds | 193,787 | 187,779 | 874 |

Every output word matches the corresponding compiled baseline. Measured cycle
reductions are approximately 4.62%, 0.19%, 1.87% and 0.20%. Small gains do not
establish profitability across workloads. The feature stays opt-in behind
`--matrix-vpu-transport`, requiring immutable prepacking and an explicit numeric
execution policy. Unknown effects, fanout and observed intermediate values retain
the original boundary.

The composed case retains one shared packing group in both variants. The
candidate changes both resulting private matrix/add edges to BF16 transport,
including their encoded-activation consumer ABIs; all874 words match and the
additional reduction is3.10%. This checks that the two transformations compose.

For the aligned160-row case the issuer writes BF16 directly to the owned logical
destination. Padded cases crop a physical BF16 readout before the VPU consumer.
The mixed consumer preserves the original FP32 conversion's NaN refusal instead
of silently accepting NaNs. Native tests check all65,282 non-NaN BF16 codes,
including infinities and signed zeros, plus positive/negative NaN refusals.
Separate native matrix-boundary tests check both formats, padding/crops and
direct destinations. The boundary-only native VPU observer is not a device
arithmetic oracle; the complete sections above ran on the actual pinned Spike
extension and checked every result word.

The real section captures source FP32 inputs immediately before the first MLP
down-projection and its residual addition. All first-layer prefix parameters and
the embedding are checked against the matched capture checkpoint. This is not a
quantized intermediate dump, a complete model, or an autoregressive decode
trajectory. These results establish no independent whole-model quality gate or
end-to-end speedup.

All comparisons use the unchanged stage-forward timing interval, including
packing, padding, dispatch, matrix/VPU work, host conversion and allocation.
Initial harness setup and console output remain excluded. Profile, checkpoint,
inputs, numeric policy, extension and simulator settings match within each pair:
`both`, `e4m3_vpu`, `mem.host_tracking=0`. Cycles are modeled, not hardware results.
Each comparison retains allocation site estimates, ELF sizes, counters and
private-edge records. Allocation estimates are not measured live memory peaks.
Original fallback entries remain in the catalog, so smaller variant workspace
does not imply a smaller reserved shared workspace.

For160 rows, the static allocation site estimate falls48,759,026→48,103,666 B.
The BF16 adapter's working workspace is931,328 B versus1,586,688 B for the
ordinary adapter, while the catalog reserves the original shared maximum.
The candidate ELF grows13,624 B. Commands remain10,661 (40 VPU), device load
bytes41,680,896 and scale bytes1,261,568. For one row, allocation estimates
fall46,153,970→46,149,874 B, the reserved/working workspace remains520,960 B,
and the ELF grows5,408 B. Commands remain1,465 (1 VPU), device load bytes11,902,976
and scale bytes371,712. These extension counters retain their wider scope;
they do not measure host conversion traffic with host tracking disabled.

`executions.tar.xz` contains all10 actual admitted ELFs, complete logs, comparisons,
captures excluding large real FP32 checkpoint files (identities retained), source
receipts and generated C/MLIR. `python verify.py` reopens every word, executable,
recipe and accepted structural admission record without extraction. This
structural admission supplies digest/declared-ISA custody, not instruction policy.

To reproduce from an editable task, select the same toolchain environment used
by the neighboring shared-activation archive's reproduce.sh, then run
`bash docs/evidence/matrix_transport_20261018/reproduce.sh` with fresh
`MX_REPRO_DIR` and the original `MX_SOURCE_STAGE`. The script captures and
compares all five pairs above. Capture real sections individually
with `python -m tools.capture_tinyllama_residual --stage MATCHED_CAPTURE --out-dir
FRESH --rows 160 --profile PROFILE` (repeat with1 row). Build each capture twice
using `tools.build_merlin_stage`, immutable prepacking, entry projection, the
capture's policy and structural admission; enable `--matrix-vpu-transport` only
for the candidate. Compare with `tools.compare_shared_preparation`. Independent
tail and broadcast cases use `tools.capture_matrix_residual_probe`, optionally
`--rows 1 --bias-broadcast`. The composed case uses `--shared-projections` and
enables shared packing in both builds. Keep generated products in the task's `.tmp`.
