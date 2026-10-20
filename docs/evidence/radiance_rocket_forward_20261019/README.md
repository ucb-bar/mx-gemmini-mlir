# Complete Radiance-derived Rocket TinyLlama forward

This comparison executes all 22 decoder layers and the LM head for the same
160 token IDs and TinyLlama checkpoint as the published compiler control.
Both run on the pinned Rocket MX+VPU Spike configuration. The
[port guide](../../radiance_rocket_model_baseline.md) records the source
revision, numerical changes, target adaptation, timing boundary and commands.

| Complete token-ID-to-F32-logits execution | Modeled cycles |
| --- | ---: |
| Radiance-derived Rocket port | **7,082,331,012** |
| Compiler control | 10,326,818,367 |

The compiler takes **45.81% more cycles**. This ratio is unqualified because
the compiler fails the complete-160 mean-KL gate. The late-33 gate passes.
These observations do not reproduce Muon execution or establish the fastest
possible handwritten Rocket implementation.

| All 160 positions against the same floating reference | Port | Compiler |
| --- | ---: | ---: |
| Mean KL, nats | **0.0774637** | 0.0821650 |
| Worst-row KL | 2.7480708 | **1.9150298** |
| Matching top tokens | 132/160 | **133/160** |

The allowed source-relative increase in mean and worst KL is 5%, with no
decrease in matching top tokens. The compiler's mean KL increases by 6.07%.
The reader checks every one of the 5,120,000 logits on both sides. It does
not infer general model quality or perplexity from this fixed input.

The port executes all expected accelerator work: 41,308,160 matrix tiles,
66,656 native loops, 116,160 VPU exponential commands, 3,798,532,096 load
bytes and 115,662,848 scale bytes. The compiler executes 40,970,240 matrix
tiles; the port's extra 0.83% comes from its padded attention key dimension.
The lower port time is not explained by omitting those matrix products.

## Diagnostic profiles

The port profile preserves every retained logit and residual tensor. The
compiler profile preserves every published compiler output word. Their
instrumented totals are 7,075,720,460 and 10,327,121,288 cycles respectively.
Image placement and instrumentation affect timing; these totals do not
replace the clean execution times above.

| Port phase group | Diagnostic cycles |
| --- | ---: |
| Softmax and probability packing | 2,434,973,774 |
| SiLU, gate/up multiplication and packing | 1,607,658,132 |
| Two per-layer normalization phases | 1,292,731,638 |
| RoPE and cache packing | 560,451,624 |
| Attention-output packing | 398,074,926 |

The compiler profile attributes 3,026,815,338 cycles to host operations,
3,651,903,335 to VPU endpoints, and 2,887,177,158 to matrix endpoints.
Endpoints include their wrapper and preparation work. Compiler softmax is
host arithmetic in this control. The source phases have different boundaries;
dividing these categories does not yield individual-kernel speedups.

## Audit

```sh
python docs/evidence/radiance_rocket_forward_20261019/verify.py
```

`forward.tar.xz` retains the complete source observations, generated program,
port C source, execution receipt, floating-reference logits and comparison.
`profiles.tar.xz` retains both actual diagnostic logs, their operation/phase
tables, source bindings and analysis. Every member is hash-bound in
`artifacts.json`. The compiler control is reopened through its existing
archive and all dependency archives.

The reader independently checks completion, guards, full observed tensors,
descriptor-derived accelerator work, numerical metrics, and unchanged profile
outputs. The large executables, checkpoint matrices and toolchain binaries
remain external build inputs. Executable identities and structural admission
are retained in the receipts; the portable archive does not contain the ELFs
or establish a hermetic rebuild. The actual source ELF and generated region
data were reopened by the matched-comparison checker before this archive was
created. These are modeled cycles, with host memory tracking disabled, rather
than FireSim or wall-clock performance.
