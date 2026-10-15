# SmolVLA all-FP8: compiler versus Nicolas MX matrix issuers

The captured SmolVLA session runs one prefix encode, ten recurrent flow
denoise steps, and one action decode on Nicolas's MX+VPU Spike. Both linked
programs use the same captured inputs, checkpoint weights, direct E4M3
matrix precision, host graph, VPU objects, profile, and timing settings
(`e4m3_vpu`, `mem.host_tracking=0`). Only the prefix and flow MX catalog
objects change: the control issues matrix commands through Nicolas's
`mx_native.h` route.

| Stage | Compiler cycles | Nicolas issuer cycles | Compiler overhead |
| --- | ---: | ---: | ---: |
| Prefix encode | 72,012,507,511 | 71,273,341,439 | 739,166,072 |
| Flow denoise, ten steps | 108,207,126,169 | 108,178,057,422 | 29,068,747 |
| Action decode | 4,235 | 4,235 | 0 |
| **Full 12-step session** | **180,219,637,915** | **179,451,403,096** | **768,234,819 (+0.428%)** |

The [comparison receipt](comparison.json) checks all 12 timed steps and 33
equal output hashes. Its changed-object list contains only the two MX
catalogs. The compiler run and link receipts are byte-identical to the
[all-FP8 full-action archive](../smolvla_all_fp8_full_action_20261014/README.md),
which reports final-action cosine 0.988524 against the captured FP32
trajectory. Run `python -B verify.py` to recheck the pair without the
multi-gigabyte checkpoint or ELF.
The copied receipts retain absolute paths from the measured build; the
verifier uses the archived receipts and logs.

This measures the matrix issuer cost inside a full captured model session.
It does not compare two independently handwritten host/VPU programs.
