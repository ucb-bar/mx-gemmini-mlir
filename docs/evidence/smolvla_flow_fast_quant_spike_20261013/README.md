# SmolVLA flow stage: exact BF16 quantizer lookup

The same captured `flow_denoise` stage and checkpoint as the
[weight preparation run](../smolvla_flow_prepack_spike_20261013/README.md)
ran on the pinned MX+VPU Spike model. Both builds prepacked the same 116
fixed RHS operands. The new build uses a lookup for live BF16-to-MX element
conversion; the existing C quantizer remains the reference implementation.

| Quantizer | Build hash | Modeled cycles | Relative time |
| --- | --- | ---: | ---: |
| C reference | `12e9345b6696` | 6,659,754,454 | 1.00× |
| Exact lookup | `a0f69bb24b2c` | 3,291,395,119 | 2.02× faster |

The lookup saves 50.6% of modeled cycles for this stage. Both runs have
identical hashes for all three outputs, zero memref rank mismatches, 555,735
MX commands, 24,170 loops, 5,444 fences, 479,784 mesh tiles, and 4,252 VPU
commands. The lookup build's ELF SHA-256 is
`77b8d53d4fdc12e5caae02a78f6a4d19bbedd37545143778de0509b9ff796a5a`.
An independently rebuilt ELF has the same SHA-256; its catalog records the
lookup header SHA-256
`6fc81f4e52567b83f4a011af13c0a5603a967448b34d8c49d2bfd3f160f1c9db`.
The [guest log](spike.log) and [extension report](gemmini_perf.txt) contain
the raw result. Spike used `GEMMINI_MODE=both`,
`GEMMINI_PERF_CONFIG=e4m3_vpu`, and `GEMMINI_PERF_SET=mem.host_tracking=0`.

The fast path is checked exhaustively against the C reference across BF16
mantissas and representative scale exponents and by packed operand byte
parity tests. This run establishes same-input compiler output parity. It does
not establish agreement with an independent quantized PyTorch reference or
full-session performance.
