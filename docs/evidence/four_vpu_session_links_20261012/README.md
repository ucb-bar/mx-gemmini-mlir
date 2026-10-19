# Four checkpoint sessions linked with MX+VPU (2026-10-12)

Merlin's multi-program session generator and the MX target linker combined
the eight device-bearing stage objects and SmolVLA's host-only action decoder
into four standalone RV64 Spike ELFs. Each link verified the captured stage
ABIs, source hashes, exact MX catalog coverage, shared checkpoint bytes, and
state handoffs. `index.json` records the receipt and ELF identities without
local filesystem paths. One copy of each session's weight blob is materialized
from the capture by Merlin's target-neutral session helper; the MX linker
packages it for the selected RV64 target.

| Session | Programs and steps | Static MX contractions | State bindings | ELF bytes |
| --- | --- | ---: | ---: | ---: |
| TinyLlama | prefill 1, decode 32 | 200 + 200 | 2 | 4,424,776,880 |
| DeepSeek Qwen 1.5B | prefill 1, decode 32 | 254 + 254 | 2 | 7,168,224,712 |
| Gemma 2 2B | prefill 1, decode 32 | 236 + 236 | 2 | 12,953,373,488 |
| SmolVLA | prefix 1, flow 10, action 1 | 243 + 148 + 0 | 3 | 917,470,016 |

The contraction counts are per distinct program graph; recurrent decode and
flow steps invoke those compiled programs again. Every device stage reports
zero operations left that match this profile's MX/VPU selectors. The
[stage coverage archive](../mx_vpu_model_arithmetic_20261011/README.md)
contains the per-family counts and standalone the pinned Spike model arithmetic tests.

The link receipts have status `linked_experimental_numeric_qualification_pending`.
A linked ELF establishes compiler and ABI coverage, not a successful inference
session. New VPU-enabled full-session runs on the pinned Spike model and an independent
quantized PyTorch model reference are required before claiming numerical
correctness or performance parity with the reference implementation’s handwritten kernels.
