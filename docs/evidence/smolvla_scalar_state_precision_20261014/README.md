# SmolVLA scalar-state precision control

The selected MX+VPU profile runs large vector operations through the VPU's
BF16 arithmetic. SmolVLA's one-element FP32 timestep update was also routed
there. Although its ABI was FP32, the earlier [full action run](../smolvla_full_action_values_20261014/README.md)
matches ten successive BF16-rounded timestep updates, ending at
`-0.0048828125` instead of the captured FP32 value near zero.

The selector now leaves one-element FP32 operations on the host. This rebuild
uses the same captured flow MLIR, MX profile, 148 matrix bindings, and weight
hash as the [isolated step-9 control](../smolvla_reference_step_replay_20261014/README.md).
The VPU source-region set loses exactly `add_96#add0` (352 to 351 regions);
the number of distinct VPU elementwise kernels falls from 19 to 18.

The linked one-step program was run on the reference implementation’s `e4m3_vpu` Spike extension
with the captured FP32 step-8 flow state, FP32 prefix cache, and step-9
timestep. The timestep output is now **`-7.450580596923828e-08`**, exactly
the FP32 source update. The 1,600-element flow output is byte-identical to
the earlier isolated replay and has **0.907116 cosine** to the captured FP32
flow output. This fixes scalar-state precision but does not close the local
flow-output numerical gap. A corrected *full* SmolVLA session has not yet
been run, so its final action quality cannot be inferred from this control.

The new isolated run measured 11,468,377,814 Spike cycles. The earlier
isolated binary measured 41,133,937,106 cycles, but a single route removal
also changed the generated host object and one extra output was dumped in
the new run. These numbers are recorded for diagnosis; this archive does not
claim a full-model speedup or attribute the entire difference to the scalar
VPU instruction.

Run `python -B verify.py` here to check the archived catalogs, link and Spike
receipts, output hashes, exact timestep, and relation to the preceding
isolated replay. `archive.py` records the source files and creates the
portable receipt after a completed Spike run.
