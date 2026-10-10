# Nicolas FP6 resident chain on Spike

Nicolas's `gemmini-mx-cleanup` source runs two E3M2 64³ or 128³ matrix
contractions under `MxGemminiRocketConfig`. Each 4-bit matrix nibble indexes a
per-group 16-entry FP6 LUT. MM1 writes packed C1 indices and E8M0 scales to
scratchpad; MM2 reuses both without a DRAM reload. MM2 loads B2 and its scales,
uses MM1's output LUT as its activation LUT, and writes C2.

The path starts with two PyTorch `matmul` sites captured by pinned model2MLIR
(`e9ded36`). This establishes the graph shape and site IDs. The frontend's
reviewed FP6 codebook selects the sites; its random PyTorch tensors and
codebook are not used as hardware data. The exact A/B/B2 packed bytes, all five
complete LUT banks, E8M0 scales, and C1/C2 goldens come from Nicolas's checked
C header. The source and header hashes bind the typed MLIR module, and the
runtime input digest covers all eleven input buffers.

The typed graph uses `runtime_lut` operations for MM1's B1, A1, and C1 banks
and MM2's B2, C1, and C2 banks. `resident_contract` names the C1 bank as MM2's
activation LUT. The connected lowerer checks the SSA tensor edges, both
sites, LUT buffer reuse, physical sizes, profile, and scratchpad placement.
It emits the source's command order, including the FP6 resident quantization
configuration after loading MM1 A-scales, and fences between packed matrix
transfers. `tools.compile_object` builds a data-free RV64 RoCC object with an
explicit fifteen-buffer ABI.

`qualify_nicolas_fp6_resident_chain.py` compiles the unmodified source and a
diagnostic driver that replaces only the accelerator issue sites. The source's
C1/C2 nibble and scale comparisons remain. Both ELFs pass on Nicolas's stock
`libgemmini` Spike extension at 64³ and 128³. Each size checks 4,096 or 16,384
FP6 codes and 128 or 512 scale bytes at **each** of C1 and C2. This qualifies
the named plain MX Spike profile; the MX+VPU configurations have separate
gates.

Reproduce either size from a checkout with the pinned RTL submodules and RISC-V
toolchain (choose `64` or `128` for `D`):

```bash
D=64
python -m tools.capture_nicolas_fp6_resident_chain \
  --matrix-dim "$D" \
  --model2mlir-root /scratch/agustin/tmp/model2mlir-mx-upstream-e9ded-20261010 \
  --mxq-root /scratch/agustin/tmp/microscaling-quant-20261009 \
  --rtl-root /scratch/agustin/tmp/gemmini-mx-cleanup-20261009 \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir "/tmp/mx-fp6-capture-$D"

python -m tools.qualify_nicolas_fp6_resident_chain \
  --matrix-dim "$D" \
  --rtl-root /scratch/agustin/tmp/gemmini-mx-cleanup-20261009 \
  --riscv-root /scratch/agustin/projects/chipyard/.conda-env/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt \
  --frontend-dir "/tmp/mx-fp6-capture-$D" \
  --out-dir "/tmp/mx-fp6-chain-$D"
```

The qualifier writes capture receipts, typed MLIR, physical commands, an
issuer object, source and compiler ELFs, patches, and Spike logs. The standalone
object emitter also accepts the typed module through `tools.compile_object` with
`--resources-dir` and `--abi-json`; the ABI file must map the eleven input slots
and four output slots in `mx_gemmini.resident_pair_buffer_map.v1`.

The [64³ archive](evidence/nicolas_fp6_connected_resident_64_266c593/index.json)
and [128³ archive](evidence/nicolas_fp6_connected_resident_128_266c593/index.json)
include the capture, connected program, packed runtime inputs, data-free
objects, source and compiler ELFs, Spike logs, and file digests. Both archives
record the byte-identical replay from a fresh clone of the published
`handwritten-implementation` commit `ef16c11`.
