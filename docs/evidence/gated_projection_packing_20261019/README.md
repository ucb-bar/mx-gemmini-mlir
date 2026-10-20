# Gate/up projections through SiLU and FP8 packing

This paired Rocket MX+VPU probe measures two FP8 projections with
M=160, N=5632, K=2048, followed by hybrid sigmoid, two FP32 multiplies
and FP8 block32 packing. Both projections retain tiles 80/128/512.
Inputs and weights are synthetic powers of two with an exact matrix oracle.

| Implementation | Control cycles | Fused cycles | Control / fused |
| --- | ---: | ---: | ---: |
| Production compiler pass | 93,842,868; 93,844,408 | 76,728,902; 76,729,906 | **1.223×** |
| Earlier deferred-issuer prototype | 93,842,868; 93,842,996 | 74,925,615; 74,926,598 | 1.252× |

The production pass reduces this sequence's mean cycles by **18.24%**.
It calls each original synchronous matrix issuer, keeps their BF16 outputs
in the existing serial workspace, and computes and packs one product row
at a time. It removes private FP32 projection readouts and intermediate
sigmoid/product tensors. The selected compiler arithmetic remains
`(gate * (1 / (1 + exp(-gate)))) * up`, with BF16 EXP and FP32 scalar operations.
It does not change weights, matrix tiles or quantized encoding.

The earlier prototype uses deferred matrix completion and separate static
row buffers. Its result is retained as a separate experiment; its larger
gain is not attributed to the production pass. Neither row establishes a
full-model speedup over `radiance-kernels`.

## Reopen the measurements

The archive contains both actual ELFs, generated C, complete input and oracle
payloads, original commands, target logs and structural admission reports.
The production C was reproduced from factory revision
`129cd185cb4b3967feb4e3b91b9445fe1779dfe0` before archiving. The manifest binds
that generated source and every archived member.

```sh
python docs/evidence/gated_projection_packing_20261019/verify.py \
  --workdir "$TASK_DIR/.tmp/gated-projection-check" \
  --out "$TASK_DIR/.tmp/gated-projection-check.json"
```

Use fresh paths. The reader requires NumPy and a host C compiler. It decodes
the actual FP8 matrix payloads, regenerates exact matrix outputs, then uses
the pinned independent VPU reference and canonical scalar quantizer to
regenerate all 929,280 expected packed bytes. It checks the corresponding
symbols in each actual ELF and every dumped byte from all four alternating
calls. Target checks include output guards and run after timing. Matrix/VPU
work counters, completion, executable identity and archive hashes also pass.
The ELF reader checks payloads; structural admission checks digest and declared
ISA. Neither alone establishes executable-code correctness.

The [target configuration guide](../../configure_target_and_precision.md#fuse-gateup-projections-through-silu-and-packing)
describes selection and fallback conditions. These are modeled Spike cycles
with host memory tracking disabled, using the same target/tool identities as
the full-model comparison. They are not FireSim measurements.
