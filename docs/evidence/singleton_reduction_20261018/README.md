# Source-exact singleton reduction placement

This is an isolated numerical diagnosis and compiler transformation check.
It does not establish a quality-qualified full quantized model speedup.

The real 160-token capture expresses position × inverse frequency as a
`[1,32,1] @ [1,1,160]` contraction. The old MX route pads K to 32 and quantizes
both operands. The independent source computes these phases on the host.
The same archived extension/profile used in the earlier full-model comparison
produces a maximum phase error of **9.234024 radians** and cosine error of
**1.999826**. Complete output dumps retain all 5,120 phase values.

Merlin's new generic singleton-reduction transformation preserves every scalar
multiply, accumulator addition, initializer, type, flag, and rounding boundary.
It removes only proven single-iteration reduction loops. Explicit MX execution
policy v2 selects host placement and accounts for these source obligations.
The original v1 behavior remains available. Production selection uses typed
operation semantics and static extents, with no model or region-name heuristic.

| Execution | Timed stage cycles | Maximum phase error | Complete values |
| --- | ---: | ---: | ---: |
| Original FP8 matrix route | 427,146 | 9.234024 radians | 5,120 |
| Transformed host route | 54,418 | 0; every FP32 word equal | 5,120 |

These are one-call stage measurements, including generated host preparation
inside the existing timing boundary. They do not measure steady-state reuse
or the full network. Output formatting occurs after timing. The mixed fixture
retains the same exact phase output and offloads a dependent, nonunit matrix
consumer: one host source obligation plus one linked MX contraction, with
complete source coverage. The host-only fixture has no MX linked-ELF admission
selection; the MX fixtures use the explicit existing structural admission
service, whose scope is digest/declared ISA rather than instruction policy.

The full 22-layer FP32 sensitivity experiment injects only the measured angles,
leaving all other computation FP32. The exact-angle injection first reproduces
every captured logit bit. Measured-angle injection gives mean KL 0.208172 and
105/160 matching top tokens; the late 33 rows give mean KL 0.287551 and 21/33.
This isolates the consequence of angle error. It is neither quantized model
execution nor an additive attribution of the original final-logit error.

`operands.npz` retains the real padded operands, native packed codes/scales,
and the operand-only approximation. That approximation is explicitly separate
from Spike's product/accumulation behavior (their maximum difference is 6).
The complete Spike logs, placement receipts, policies, and selected typed IR
are archived. ELF files, checkpoints, and full sensitivity logits are retained
in the task scratch directory rather than duplicated here.

Run the portable check from this repository:

```bash
python docs/evidence/singleton_reduction_20261018/verify.py
```

To also recompute every full sensitivity metric using the retained arrays:

```bash
python docs/evidence/singleton_reduction_20261018/verify.py \
  --full-reference ../.tmp/mx_model_goal_20261018/onepass160/capture/golden.npy \
  --sensitivity-output ../.tmp/mx_model_goal_20261018/rotary_probe/full_sensitivity_final/angle_perturbed_fp32_logits.npy
```

For a new execution, use the task capture interpreter with Torch, Transformers,
safetensors and xDSL, put the task Merlin/model2MLIR on `PYTHONPATH`, and set
`MERLIN_COMPILER_PYTHON` to the compiler interpreter with torch-mlir. Temporary
files must use the task's `.tmp/`. Starting from the verified full capture:

```bash
full_stage=../.tmp/mx_model_goal_20261018/onepass160/capture
singleton_work="$TMPDIR/singleton-reproduction"
singleton_profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
python -m tools.capture_tinyllama_rotary_probe \
  --stage "$full_stage" --profile "$singleton_profile" --out-dir "$singleton_work"
for route in matrix host; do
  GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu \
  GEMMINI_PERF_SET=mem.host_tracking=0 \
  GEMMINI_PERF_OUT="$singleton_work/${route}_perf.txt" \
  python -m tools.build_merlin_stage \
    --stage "$singleton_work/capture" --workdir "$singleton_work/build_$route" \
    --profile "$singleton_profile" --merlin-src ../merlin/src \
    --llvm-install /scratch/agustin/projects/oscar-merlin/third_party/llvm-install \
    --riscv-root /scratch/agustin/projects/chipyard/.conda-env/riscv-tools \
    --execution-policy "$singleton_work/${route}_policy.json" \
    --structural-elf-admission --spike-arena-base 0xd0000000 \
    --spike-extlib docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so \
    --output-dump-cap 5120
done
```

Use a fresh output directory and `--include-consumer` for the mixed fixture.
The archived `capture_probe.py`, `capture_mixed_probe.py`, and
`native_operand_probe.py` record the original experimental scripts; the CLI
above provides the configurable capture route. Full sensitivity reproduction
uses `tools.diagnose_tinyllama_rotary_angles` with the verified Spike angle
array, full capture, and a fresh output directory.

Remaining: complete and compare the baseline and optimized 22-layer Spike
forwards under the unchanged source-relative quality gates, then evaluate
cached decoding and the other optimization areas in the active goal.
