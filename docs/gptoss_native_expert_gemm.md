# Native gpt-oss expert GEMM source stage

`tools.emit_gptoss_native_expert_gemm` joins two independently selected inputs:
the original checkpoint's packed MXFP4 B tile and the user's `host_c_v1`
activation quantizer. It emits one direct FP4, BF16-readout GEMM stage. The
activation callback receives the live contiguous float32 A pointer when this
stage is invoked; the BF16 A bytes in the earlier tile binder are retained
only to recheck that binder's receipt.

The tool reopens the range extractor receipt and selected safetensors slices,
recomputes the tile's Gemmini B codes/scales, and compares every byte and the
whole tile manifest. The tile must already name the same source-bound MX
profile and one legal direct FP4 A/B mode. The run recipe selects
`weight_source: preserve_native`, names the checkpoint revision, original
blocks/scales tensor names and raw SHA-256 values, and selects FP4 E2M1 direct
for both A and B. Its `external_wire.activation_runtime` names the versioned
guest C symbol and pins the C source and ABI/custom headers by SHA-256. The
compiler does not choose the numerical activation quantization rule.
The recipe should first pass npu-exploration's `model_run.v2` parser. This MX
stage checks the selected fields and hashes that it consumes; it does not
reimplement the complete npu recipe parser or invoke the Python provider.

For a managed task with the range extractor fixture and profile-bound tile:

```sh
PYTHONPATH=../merlin/src:. python3 -m tools.emit_gptoss_native_expert_gemm \
  --tile-dir ../.tmp/gptoss20b-native-tile-profile \
  --fixture-dir ../.tmp/gptoss20b-expert0 \
  --extractor-receipt ../.tmp/gptoss20b-expert0/receipt.json \
  --run ../.tmp/gptoss-native-fp4-user-run.json \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --site-id gptoss.expert0.gate_up.tile0 \
  --callback-c ../.tmp/user_activation_quantizer.c \
  --abi-header ../npu-exploration/compiler/model_activation_abi.h \
  --out-dir ../.tmp/gptoss20b-expert-gemm-stage
```

The output has the checked native B arrays, generated matrix issuer, dense
model entry, pinned user C and headers, and a receipt. The model entry uses
the existing live dense adapter, invokes the callback before Gemmini loads A,
checks callback status and invalid E8M0 scales, and records guest prologue
cycles. `tests.test_native_expert_gemm` calls the generated entry twice with
different live A values against a host issuer stub and verifies that the
original packed B pointer and both A results reach the issuer.

The site ID here is a user-named **tile fixture**, since no gpt-oss model2MLIR
capture is bound. The status is
`generated_source_only_no_model_capture_elf_or_spike_parity`. The output has
not been linked into a guest ELF or run on the matching Spike extension. The
host stub does not model MX arithmetic. After a captured source region and
matching toolchain are available, compare every BF16 output with an
independent unpack/matmul oracle before routing, sinks, or performance claims.
