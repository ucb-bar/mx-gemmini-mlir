# Encode physical zero rows without quantizing them

The explicit experimental `--virtual-zero-padding` option skips materializing
and quantizing padded activation rows. Merlin plans the remaining dense
zero/copy operations; MX writes the exact physical zero codes and scales.
The existing block rounding, group order and FP4 shared nibbles are preserved.
No input values outside the logical rows are read.

Six admitted, uninstrumented projection pairs preserve every output word:

| Logical input | Format | Explicit padding cycles | Virtual rows cycles | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 × 37 | FP8 | 74,505 | 17,428 | 76.6083% |
| 1 × 37 | FP4 | 139,674 | 17,317 | 87.6018% |
| 19 × 37 | FP8 | 232,868 | 183,441 | 21.2253% |
| 19 × 37 | FP4 | 234,655 | 183,288 | 21.8904% |
| 160 × 37 | FP8 | 1,474,814 | 1,474,814 | 0% |
| 1 × 2048 | FP8 | 2,193,323 | 349,814 | 84.0510% |

Both sides already share activation packing. Each fixture has three BF16
linear projections with widths 23, 41 and 17 and a complete concatenated F32
output. Timers include host conversion, packing and allocation. The 160-row
control produces an identical ELF: its rows are already aligned, so selection
adds no alternative path. These measurements are complete projection stages,
not whole-model speedups or independent source-quality results.

Tests cover exact codes/scales for FP4/FP8, odd/even live row counts, column and
row tails, signed zeros, subnormals and large values. Late-group nonfinite
refusals preserve all prior and untouched output bytes. Canary bytes remain
unchanged. Original MXQuant payload hashes and the fast/reference LUT checks
continue to pass.

The full compiled TinyLlama decoder retains the same 155 prepacked checkpoint
kernels, 44 sharing groups/66 removed preparations, numerical policy and host
allocation-site estimate. Its two private preparation buffers fall from
131,072 and 360,448 bytes to zero, and its shared kernel workspace falls from
1,190,912 to 1,059,840 bytes. Actual ELF `.bss` shrinks by **622,592 bytes**.
These are static compiled storage costs, not runtime peak memory or a measured
whole-session speedup. The new full 128+32 session measurement is separate.

The compact archive retains all four captures, all twelve actual ELFs,
complete native outputs, build/admission/compilation receipts, policies,
generated code and relevant source snapshots. `decoder_static_resources.json`
retains actual full-decoder ELF identities and pinned readelf observations;
the large decoder ELFs remain in the task workspace. Structural admission is
not independent hardware or forbidden-instruction-policy qualification.

Run `python3 docs/evidence/virtual_padding_20261019/verify.py` to reopen every
member, ELF/admission identity, output word/digest and cycle count. Add
`--build-root TASK_TMP/mx_model_goal_20261018` to reopen both actual large decoder
images and repeat their pinned readelf observations too. Verification uses
the standard library and does not extract temporary files.

Reproduce with `tools.capture_shared_projection_probe --bf16-projections`,
the table's rows/width and the pinned profile. The retained policies select
FP8 or FP4 consistently per pair. Build both sides with checkpoint prepacking,
unused-parameter pruning, structural ELF admission and shared activation
packing; add only `--virtual-zero-padding` to the candidate. Dump the complete
`rows × 81` outputs, then use `tools.compare_shared_preparation`. Keep the
existing numerical and timing gates. The option remains off by default.

For example, after selecting `mx_python`, `merlin_source`, `llvm_install` and
`riscv_root` in the configured task environment, reproduce the one-row FP8 pair:

```bash
export GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu
export GEMMINI_PERF_SET=mem.host_tracking=0
mx_profile=profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json
mx_extension=docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so
mx_case="$TMPDIR/virtual-padding-reproduction"
"$mx_python" -m tools.capture_shared_projection_probe --rows 1 --width 37 \
  --bf16-projections --profile "$mx_profile" --out-dir "$mx_case"
for mx_variant in current virtual; do
  mx_options=()
  if [[ "$mx_variant" == virtual ]]; then mx_options+=(--virtual-zero-padding); fi
  "$mx_python" -m tools.build_merlin_stage --stage "$mx_case/capture" \
    --workdir "$mx_case/$mx_variant" --execution-policy "$mx_case/policy.json" \
    --profile "$mx_profile" --merlin-src "$merlin_source" \
    --llvm-install "$llvm_install" --riscv-root "$riscv_root" \
    --prepack-checkpoint-weights --prune-unused-checkpoint-weights \
    --share-activation-packing --structural-elf-admission \
    --spike-arena-base 0xd0000000 --output-dump-cap 81 \
    --spike-extlib "$mx_extension" "${mx_options[@]}"
done
"$mx_python" -m tools.compare_shared_preparation --stage "$mx_case/capture" \
  --build "current=$mx_case/current" --build "virtual=$mx_case/virtual" \
  --out "$mx_case/comparison.json"
```
