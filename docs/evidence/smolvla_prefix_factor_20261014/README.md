# SmolVLA step-9 prefix-cache control

The [first isolated replay](../smolvla_reference_step_replay_20261014/README.md)
reset both the prior flow state and the prefix cache to captured FP32 values.
That raised the final flow-step cosine to FP32 from 0.054 to 0.907, but did
not distinguish those two inputs.

This control holds the **compiled step-8 flow state and compiled timestep**
fixed. It changes only the prefix cache to the captured FP32 value. The
same compiled flow object, weights, profile, and reference `e4m3_vpu` Spike
extension are used. The compiled state bytes and BF16-rounded timestep are
verified against the archived full run's output hashes before linking.

| Step-9 input | Cosine to FP32 step-9 output | Normalized RMSE |
| --- | ---: | ---: |
| Full compiled history and prefix | 0.054454 | 1.725878 |
| Compiled history, FP32 prefix cache | **0.066057** | **1.697439** |
| FP32 history and FP32 prefix cache | 0.907116 | 0.436705 |

The compiled-history/FP32-prefix output has 0.984375 cosine to the original
full compiled output. At this final step and fixed compiled state/timestep,
resetting the prefix cache changes the output little. The large recovery
when both state and prefix are reset therefore points to accumulated flow
state error as the main immediate input difference. This does not prove that
prefix errors never contributed to earlier state drift, or that the remaining
one-step error is caused by any one MX or VPU operation.

The one-step control measured 41,134,827,305 modeled Spike cycles. It has a
cold-start boundary and excludes prefix encoding and action decoding, so it
is not a full-session cycle comparison. Its output values and timing report
are archived only as a diagnostic. Run `python -B verify.py` here to check
the bound inputs, flow object, Spike outputs, and all three comparison rows.
