# Full SmolVLA action-value diagnostic

This archive contains the completed output-tap 12-step SmolVLA Spike run.
It keeps all ten flow outputs and the final action, checks each
dump against its guest digest, and compares the values with the captured
unquantized FP32 trajectory. The capture uses synthetic inputs.

| Output versus captured FP32 | Cosine | Normalized RMSE |
| --- | ---: | ---: |
| Flow step 0 | 0.997092 | 0.076213 |
| Flow step 5 | 0.629726 | 0.802661 |
| Flow step 9 / final action | 0.054454 | 1.725878 |

The iterative flow amplifies numerical drift on this capture. The final
action bytes equal the last flow output. The archive does not establish an
acceptable task-quality threshold.

The tapped executable reuses the earlier stage objects. Its complete
33-output digest list must match both the earlier compiler run and the faster
compiler side of the [Nicolas issuer comparison](../nicolas_matrix_substitution_smolvla_full_20261013/README.md).
That identity transfers the **output-value diagnostic** to the faster run;
the tapped executable has a different timing cost and does not transfer its
cycle count. The FP32 comparison measures drift, not acceptance under exact
FP4/FP8 mesh and VPU arithmetic.
The earlier optimized compiler run took **72,066,544,354** modeled cycles;
the output-tap run took **613,171,131,189** because its stage objects differ.
The tap cycle count is not used in the issuer-swap comparison or as a
full-model source-kernel speedup.

After archiving, verify all copied artifacts and recompute the action metrics:

```sh
python3 -B docs/evidence/smolvla_full_action_values_20261014/verify.py
```
