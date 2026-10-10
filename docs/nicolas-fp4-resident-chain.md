# Nicolas FP4 resident chain on Spike

The `gemmini-mx-cleanup` source
`matmul_tiled_fp4_64x64_chain.c` runs two 64×64×64 E2M1 contractions. MM1
leaves packed C1 codes and E8M0 scales on chip; MM2 reads C1 in place and
loads only its new B2 codes and scales. This source selects the plain
`MxGemminiRocketConfig`, so its compiler gate uses that profile. The separate
MX+VPU profiles remain covered by their VPU and SPAD_REQUANT gates.

The compiler path is:

1. `capture_nicolas_fp4_resident_chain.py` captures the two PyTorch `matmul`
   sites with model2MLIR and `examples/fp4-policy.yaml`.
2. `fp4_plain_chain.py` binds those sites to the packed A/B/B2 bytes and scales
   in Nicolas's checked source header. The typed graph has a live
   `contract → readout_quantized → resident_contract` edge.
3. `resident_pair_graph.py` checks SSA edges, buffer lengths, profile, and
   scratchpad lifetimes, then issues the same physical commands used by the
   source. `emit_resident_pair_object.py --precision fp4_e2m1` builds a
   data-free RV64 RoCC object with an explicit ten-buffer ABI.
4. `qualify_nicolas_fp4_resident_chain.py` links the compiler issuer into a
   diagnostic copy of Nicolas's source. Only accelerator issue sites change;
   the source's C1/C2 nibble and scale comparisons remain. Both source and
   compiler ELFs run against Nicolas's pinned `libgemmini` Spike extension.

The PyTorch capture establishes graph sites and shapes. Its random examples
are not used as hardware operands. Nicolas's packed header supplies the exact
runtime bytes and source goldens.

Reproduce from a checkout with the pinned toolchain and RTL submodules:

```bash
python -m tools.capture_nicolas_fp4_resident_chain \
  --model2mlir-root /scratch/agustin/tmp/model2mlir-mx-upstream-e9ded-20261010 \
  --mxq-root /scratch/agustin/tmp/microscaling-quant-20261009 \
  --rtl-root /scratch/agustin/tmp/gemmini-mx-cleanup-20261009 \
  --profile profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json \
  --mx-opt build/tools/mx-gemmini-opt --out-dir /tmp/mx-fp4-capture

python -m tools.qualify_nicolas_fp4_resident_chain \
  --rtl-root /scratch/agustin/tmp/gemmini-mx-cleanup-20261009 \
  --riscv-root /scratch/agustin/projects/chipyard/.conda-env/riscv-tools \
  --mx-opt build/tools/mx-gemmini-opt --frontend-dir /tmp/mx-fp4-capture \
  --out-dir /tmp/mx-fp4-chain
```

The [evidence archive](evidence/nicolas_fp4_connected_resident_266c593/index.json)
records the captured MLIR, bound graph, physical commands, data-free object,
both ELFs, and Spike logs. The pinned run matched 4,096 FP4 nibble codes and
128 E8M0 scale bytes in each of C1 and C2. This is a source and Spike
qualification for the plain MX profile; it does not qualify RTL timing or the
FPGA image.
