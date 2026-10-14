# Full SmolVLA action-value diagnostic

This archive is populated after the output-tap 12-step SmolVLA Spike run
finishes. It keeps all ten flow outputs and the final action, checks each
dump against its guest digest, and compares the values with the captured
unquantized FP32 trajectory. The capture uses synthetic inputs.

The tapped executable reuses the earlier stage objects. Its complete
33-output digest list must match both the earlier compiler run and the faster
compiler side of the [Nicolas issuer comparison](../nicolas_matrix_substitution_smolvla_full_20261013/README.md).
That identity transfers the **output-value diagnostic** to the faster run;
the tapped executable has a different timing cost and does not transfer its
cycle count. The FP32 comparison measures drift, not acceptance under exact
FP4/FP8 mesh and VPU arithmetic.

After archiving, verify all copied artifacts and recompute the action metrics:

```sh
python3 -B docs/evidence/smolvla_full_action_values_20261014/verify.py
```
