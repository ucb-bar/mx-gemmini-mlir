# Encode padded activation rows directly

Select `--virtual-zero-padding` on `tools.build_merlin_stage` to experiment with
encoding padded matrix rows directly. Selection follows the current logical and
physical matrix dimensions. A row-aligned matrix retains its original packing
path, including column padding when needed. The option is off by default.

Merlin owns the dense padding plan and remaining host zero/copy loops. The MX
consumer explicitly implements positive-zero row extension into its packed
activation ABI. Only live rows undergo BF16 rounding, block maximum reduction
and element quantization. Padded rows receive the ordinary zero code and its
scale derived from the existing quantizer rule. FP4 pairs preserve the live low
nibble when the first padded row is odd. Group-major order preserves the exact
refusal and partial-write behavior of explicit padding.

The source is dense `live_rows × physical_k`; only those input floats are
accessed. When columns require padding, Merlin materializes those columns for
the live rows. With aligned columns the original live input is consumed directly.
Codes and scales still cover the complete physical matrix, and matrix schedules,
operand rounding and completion remain unchanged.

Dense adapter generation uses Merlin's `llvmlower.dense_padding`; select the
compatible installed Merlin package or the configured Merlin checkout. Target
instructions, nibble layout, scale encoding and the packed-pointer ABI stay in
this repository. The catalog and stage receipt retain the explicit option.

Compare identical capture, policy, checkpoint and simulator settings. Keep
`--share-activation-packing` consistent on both sides if selected, add only
`--virtual-zero-padding` to the candidate, dump the complete output and run
`tools.compare_shared_preparation`. A projection-stage win does not establish
full-model or cached-trajectory performance or numerical qualification.
