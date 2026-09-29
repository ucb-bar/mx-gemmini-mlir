# MX Gemmini Phase 0 review

Review snapshot: 2026-09-29. The selected software contract remains
`unreviewed`. This review applies to the exact Gemmini
`f0167390b56fb315deea90ac1fc3983772e92d82`, MxGen
`a27ce3cd81513210c21f971ec3977defd13fa21e`, and
`GemminiMxFPConfigs.standaloneMxFPConfig`. The RTL at those revisions is the
authority for the selected configuration.

## Results

| Area | Result | Boundary |
| --- | --- | --- |
| Source selection | `rtl_check` matches the selected revisions, the seven pinned source files, format codes and widths, max finite values, BF16 readout code, block-scale floor 104, and the selected rounding paths. | Source cross-check; not a complete elaboration closure. |
| Operand conversion | Differential comparison against [`microscaling-quant` at `ea3f5bb`](https://github.com/chloe-wong/microscaling-quant/tree/ea3f5bb) used RNE, the 2^-23 hardware scale floor, and E3M1 before FP4. For each of MXFP8, MXFP6, and MXFP4, the 65,280 finite BF16 bit patterns had identical E8M0 scales and numerical BF16 dequantized values. Random, zero, and edge-value blocks also matched. | One FP8 and one FP6 negative-zero sign differed from the independent model; FP4 had 288 zero-sign differences and no nonzero differences. The RTL's `E3M1Tofp4` canonicalizes zero to positive zero. These comparisons do not check mesh arithmetic or all block combinations. |
| Capture handoff | Mixed FP8/FP6/FP4 sites and one resident output chain pass model2MLIR capture and the MX dialect verifier. | This checks identity and operation structure, not executable lowering or whole-model accuracy. |
| Numerical review | The fake-quant kernel models BF16 operand conversion. | Mesh product truncation, the 16-lane reduction schedule, FP6 LUT choice, nonfinite poisoning, output requantization, host operations, and full-model accuracy still need selected-oracle evidence. |

The current [`gemmini-mx-cleanup` head at `2029218`](https://github.com/ucb-bar/gemmini/commit/2029218197f771ce71416f859d975bea47b7aabc)
is seven commits after the selected pin and selects MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`. It adds a separate E4M3
single-format configuration. It also changes the shared controller and loop
paths: scale-load funct 27 now has pitch, destination, row-count, and gating
fields; loop-managed scale commands 31 and 32 were added; requantizer LUT
construction is conditional. The selected standalone configuration still has a
LUT, but its existing simulator and source-bound diagnostics cannot qualify
these changed shared paths. The TorchAO kernel now rejects a contract retargeted
to another Gemmini revision, MxGen revision, or configuration until reviewed.

[`pi0-quant`](https://github.com/chloe-wong/pi0-quant) and
[`smolVLA-quant`](https://github.com/chloe-wong/smolVLA-quant) currently describe
per-tensor power-of-two FP8 model experiments. They are useful for workload
coverage and accuracy questions, but do not establish this contract's per-32
MX scale rule or its FP6/FP4 behavior. `microscaling-quant` supplies the
operand comparison above; its documented hardware-matched systolic result is
for MXFP8 E4M3, not a qualification of every format and model here.

Reproduce the operand comparison with a clean checkout of the stated
`microscaling-quant` revision in the same Python environment as this package:

```sh
python -m mx_gemmini_support.rtl_check /path/to/pinned-gemmini
python -m mx_gemmini_support.review_operands /path/to/microscaling-quant
```

The source check rejects a changed Gemmini or MxGen revision or any changed
pinned source file. The operand comparison rejects a changed reference revision
or locally changed `mxq` source and prints per-format, per-case counts.

## Phase 0 admission work

1. Freeze the intended RTL revision. For `2029218`, rebuild and bind the
   standalone elaboration and simulator, audit the new scale-load/loop protocol,
   update source and toolchain pins, and rerun the format and chain diagnostics.
   Keep the existing `f016739` receipts attributed to their original revision.
2. Select exact MX application captures and a versioned site policy. Review any
   FP6 codebooks, host placement, and accuracy choices with the numerical owners.
   The example FP8-only policy is a bringup policy, not a reviewed whole-model
   selection.
3. Derive a new MX-only conformance requirement from those captures. The retained
   synthesis sidecar is `unverified_legacy` and contains BF16/int8 accelerator
   cells outside this three-format contract. Supply a same-target capability
   contract and selected RTL facts, then regenerate and review a digest-bound
   synthesis profile under the run artifact root.
4. Run Merlin Phase 0 preflight on those selected inputs, then generate capsules
   and check independent L0/L1 numerics plus source-bound Spike L2 and Verilator
   L3. Review host transfers and the complete census before admitting a corpus.

The current `mx-gemmini-functional --phase 0` preflight reports
`configuration_ready: false` because the selected synthesis profile is
`unverified_legacy`. No Phase 0 admission or functional compiler certificate is
claimed by this review.
