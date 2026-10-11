# Four checkpoint-backed model stage links

The [summary](summary.json) records nine receipt-bound RV64 stage ELFs built
from model2MLIR captures through Merlin's whole-model host lane and the MX
catalog. These are the full pretrained TinyLlama, DeepSeek-R1-Distill-Qwen-1.5B,
Gemma 2 2B, and SmolVLA stage graphs, with synthetic captured inputs. No
checkpoint bytes or binaries are stored in this archive.

Every selected MX contraction has an exact source binding and a compiled
FP4 or FP8 issuer. FP4 covers ordinary matrix work; FP8 covers the selected
attention matrix work. SmolVLA action decode has no selected MX contraction
and links through Merlin's host lane. Other operations, including the current
f32 rank-three softmax regions, remain in the compiled host lane. The chosen
profile is Nicolas's `MxE4M3Fp4VpuGemminiRocketConfig` at SHA256
`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`.

The summary preserves each source MLIR digest, the build hash, per-stage MX
placement counts, and ELF size from the local build receipts. It is
**compilation evidence only**. A linked stage does not establish that the
full recurrent session runs, that VPU operations execute, or that FP4/FP8
outputs match an independent quantized reference. The captured sources were
generated on the model2MLIR task branch based on `02c61c2`; that branch later
merged upstream `d164bf0` without recapturing these receipt-bound artifacts.

The [full-model compilation notes](../../full_model_compilation.md) describe
the session linker, Spike verifier, and remaining numerical and VPU gates.
