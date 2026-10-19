# SmolVLA full action with FP32 scalar timestep

This archive records one captured SmolVLA session on the reference implementation’s MX+VPU
Spike extension: one prefix stage, ten recurrent flow steps, and one final
action stage. The flow stage updates its scalar timestep in FP32. The
checkpoint, target profile, program roster, prefix object, and action object
match the earlier [mixed FP4/FP8 run](../smolvla_full_action_values_20261014/README.md).
Only the flow stage object changes.

All ten dumped scalar timesteps match the captured FP32 recurrence exactly.
The final action has **0.058102 cosine** and **1.830469 normalized RMSE**
against the captured unquantized FP32 action. The prior scalar route had
0.054454 cosine. The corrected output-tap guest reports
**316,515,603,374 modeled cycles**. With the same output taps, each flow
step falls from about **41.13 billion** modeled cycles on the prior scalar
route to **11.47 billion** on the FP32 host scalar route. The prefix and
action step counts are unchanged. This measures the cost of the removed
one-element VPU path on this Spike model; the output values differ, so it
is not an equal-output speedup. The fast issuer-swap run also has a
different instrumentation boundary.

The action drift persists after the timestep correction. This is a numerical
diagnostic on one fixed synthetic capture, not a robotics task-quality
threshold. The archived [FP32 comparison](fp32_comparison.json) gives all
ten flow-step metrics; the [receipt](receipt.json) binds their output bytes
to the completed Spike run.

Verify from the repository root:

```sh
python -B docs/evidence/smolvla_scalar_full_action_20261014/verify.py
```
