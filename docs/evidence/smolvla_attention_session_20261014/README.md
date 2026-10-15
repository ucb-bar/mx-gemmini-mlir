# SmolVLA recurrent attention operands and late source replays

The pinned model2MLIR SmolVLA flow stage has **10 recurrent denoising steps**.
Each step has 16 attention calls: eight with 163 keys and eight with 113 keys.
Each call has 15 heads and 50 query rows of width 64. The
[session manifest](session_capture_manifest.json) records all **160 calls and
2,400 head instances**, their QK → mask → PV order, output checks against
the captured FP32 trajectory, and hashes of the ten operand archives.
The first-step Q/K/P/V/mask values for both shapes match the previously
[qualified cutpoints](../smolvla_attention_cutpoint_20261014/README.md)
bit for bit. The complete ten-step operand archives are reproducible from
the pinned model and capture; they are not committed because they total about
229 MB.

The final-step [163-key](call14/cutpoint.npz) and
[113-key](call15/cutpoint.npz) head-0 operands are archived here. Nicolas's
fused MX+VPU flash source, with the same checked Boolean-mask adaptation as
the first-call qualification, ran both on the pinned Spike extension with
`e4m3_vpu` timing. Its serial and pipelined outputs match in all 4,096 BF16
positions per padded head. The source logs and file hashes are checked by
[`verify.py`](verify.py).

| Final flow step, head 0 | Pipelined modeled cycles | Error vs captured FP32 attention |
| --- | ---: | ---: |
| Call 14, 163 keys | 16,955 | 5.3538% relative Frobenius |
| Call 15, 113 keys | 15,163 | 3.9587% relative Frobenius |

These are independent **attention cutpoints**, not a stitched SmolVLA stage or
whole-model cycle comparison. The model's other operators, all attention
heads, prefix stage, and action decoder still need one connected source run
before a full source-versus-compiler ratio is valid.

To regenerate the ten operand archives, use the same pretrained checkpoint
and model2MLIR flow capture as the full-model compilation guide:

```bash
python capture_session.py \
  --model2mlir-root <model2MLIR-checkout> \
  --capture-root <flow_denoise-capture-directory> \
  --out-dir <scratch-output-directory>
python verify.py --step-dir <scratch-output-directory>
```

`capture_session.py` loads the checkpoint locally, advances the flow state and
timestep through all ten steps, and checks each result against the captured
FP32 action trajectory. `verify.py` without `--step-dir` checks the archived
final-step source replays; with it, the verifier also checks every regenerated
operand archive and the final-step Q/K/P/V/mask identity.
