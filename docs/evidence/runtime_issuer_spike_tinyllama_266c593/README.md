<a id="live-encoded-pointer-mx-issuer-on-nicolass-spike"></a>

# Live encoded-pointer MX issuer on the pinned Spike model

The [receipt](receipt.json) records a source-bound 16×32×2048 FP8 TinyLlama
projection from the archived model2MLIR graph. `tools.qualify_runtime_issuer_spike`
builds one data-free issuer and calls it twice in one standalone RV64 executable.
The second call uses copied packed tensors and scales at different addresses.
Both 512-element BF16 outputs match the archived independent MX golden, and
Spike exits zero. This tests the runtime pointer boundary and repeated issue;
it does not test runtime quantization, Merlin's dense tensor ABI, or a whole
model.

Reproduce from this checkout with the pinned `gemmini-mx-cleanup` RTL and its
submodules at `266c593`:

```sh
python -m tools.qualify_runtime_issuer_spike \
  --rtl-root /path/to/gemmini-mx-cleanup-266c593 \
  --riscv-root /path/to/pinned/riscv-tools \
  --out-dir /new/output/directory
```

The command writes build logs, the ELF, a Spike log, and a fresh receipt to
the output directory. Compare its issuer, ELF, and Spike log hashes with the
checked-in receipt. The resulting object uses packed FP8/E8M0 pointers; it
cannot yet be linked to an unmodified model2MLIR `f32` contraction.
