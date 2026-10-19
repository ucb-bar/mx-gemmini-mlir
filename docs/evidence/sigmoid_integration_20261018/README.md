# Source-connected sigmoid integration

The optional `--connect-vpu-sigmoid-multiplies` stage flag connects a selected
sigmoid and two following full-shape multiplies. It requires an explicit
execution policy, prepacked weights and `--connect-vpu-multiplies`. It remains
an experiment with no automatic profitability claim.

Merlin proves privacy, input membership and legal movement of the pure source
arithmetic across independently pure work or declared read-only destination
calls. The MX provider checks the actual strict scalar bodies, selected site
policy, layout and resources, preserves each original BF16 rounding boundary,
and supplies the resident kernel and original-call fallback. The rewrite runs
before ordinary unary/binary lowering and accounts for all three source
obligations against the exact compiled catalog. It does not reorder existing
device calls. Operand order has separate endpoints so exceptional fallback
retains the original order.

An independent matrix-tail fixture computes
`(x * sigmoid(x)) * linear(x)` for a 19×37 input and 37×37 checkpoint weight.
The source activation spans an independent matrix operation. Both actual
admitted compiler ELFs use the same capture, FP8 policy, target and simulator:

| Complete stage | Modeled cycles | Output |
| --- | ---: | --- |
| Existing multiply connection | 197,956 | All703 words retained |
| Sigmoid plus multiplies | 174,366 | All703 words bit-identical |

The stage interval improves by 11.92%. Extension-wide load traffic decreases
from 14,848 to 13,440 bytes; the VPU still executes six commands. Traffic
counters include work outside the measured stage interval. Commands decrease
from 80 to 77 while fences increase from 81 to 83. The provider reports a
198,016-byte scratchpad high-water mark against 262,144 available bytes.
Linked BSS increases from 42,888 to 47,112 bytes; the archive retains the
generated kernel's fallback arrays and host buffers. This comparison establishes
transport parity to the compiled control, not FP32 model quality.
The immutable input capture, complete consoles, ELFs, admissions, recipes,
generated code and source-obligation receipts are retained in the archive.
`python verify.py` verifies them using the standard library.

The earlier [standalone qualifier](../sigmoid_transport_20261018/README.md)
also records exhaustive bounded BF16 gates, tails, saved MLP operands and an
unprofitable late fallback. That regression remains relevant to selection.
The typed 160-token TinyLlama graph exposes 22 legal chains of shape1×160×5632.
Its full-model simulation is pending; neither this fixture nor recognizing
those chains establishes a whole-model win. No numerical gates changed.

Reproduce with `bash reproduce.sh FRESH_OUTPUT_DIRECTORY` from the repository
root using the configured Merlin/compiler environment. The two build commands
include full output dumps, explicit structural ELF admission and identical
timing boundaries. The archive uses the final operand-order-specific endpoint
names; an earlier exploratory build is not included.
