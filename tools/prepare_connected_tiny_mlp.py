"""Capture a TinyLlama MLP multiply section from an existing checkpoint layer bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-bundle", required=True, type=Path)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layer-index", type=int, default=5)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory must be fresh")
    sys.path.insert(0, str(args.model2mlir_root.resolve()))
    import numpy as np
    import torch
    import m2m
    from m2m.coverage import opaque_report
    from safetensors.torch import load_file
    from transformers import AutoConfig, DynamicCache
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    if Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root.resolve():
        raise ValueError("model2MLIR resolved to another checkout")
    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_files_only=True)
    config._attn_implementation = "eager"
    layer = LlamaDecoderLayer(config, args.layer_index).eval()
    weights = load_file(args.layer_bundle / "weights.safetensors")
    layer.load_state_dict({key.removeprefix("layer."): value for key, value in weights.items()}, strict=True)
    with np.load(args.layer_bundle / "inputs.npz", allow_pickle=False) as stored:
        inputs = tuple(torch.from_numpy(stored[f"in{i}"].copy()) for i in range(6))
    grabbed = {}

    def record(name):
        def hook(_module, _args, result):
            grabbed[name] = result.detach().clone()
        return hook

    layer.mlp.gate_proj.register_forward_hook(record("gate"))
    layer.mlp.up_proj.register_forward_hook(record("up"))
    with torch.no_grad():
        cache = DynamicCache()
        cache.update(inputs[1], inputs[2], args.layer_index)
        actual = layer(inputs[0], attention_mask=inputs[5], past_key_values=cache,
                       position_embeddings=(inputs[3], inputs[4]))
        if isinstance(actual, tuple):
            actual = actual[0]
        reference = torch.from_numpy(np.load(args.layer_bundle / "golden.npy", allow_pickle=False))
        if not torch.allclose(actual, reference, rtol=1e-5, atol=1e-5):
            raise ValueError(f"checkpoint layer differs from captured golden: {(actual-reference).abs().max().item()}")
        section_inputs = (grabbed["gate"], torch.sigmoid(grabbed["gate"]), grabbed["up"])
    del layer, weights, actual, reference, inputs, cache
    gc.collect()

    class MultiplySection(torch.nn.Module):
        def forward(self, gate, sigmoid, up):
            return (gate * sigmoid) * up

    result = m2m.convert(MultiplySection().eval(), section_inputs,
                         backend="fx_importer", capture_trace=True)
    if not result.ok or opaque_report(result.mlir_text):
        raise ValueError("MLP section capture did not produce complete typed MLIR")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "source.mlir").write_text(result.mlir_text)
    (args.out_dir / "capture_trace.json").write_text(json.dumps(result.capture_trace, indent=2) + "\n")
    with torch.no_grad():
        golden = MultiplySection()(*section_inputs)
        proxy = ((section_inputs[0].bfloat16() * section_inputs[1].bfloat16()).bfloat16()
                 * section_inputs[2].bfloat16()).float()
    np.savez(args.out_dir / "inputs.npz", **{f"in{i}": value.numpy() for i, value in enumerate(section_inputs)})
    np.save(args.out_dir / "golden.npy", golden.numpy())
    np.save(args.out_dir / "bf16_policy_golden.npy", proxy.numpy())
    receipt = dict(schema="mx_gemmini.connected_mlp_capture.v1", layer_index=args.layer_index,
                   model2mlir_revision=subprocess.check_output(
                       ["git", "-C", str(args.model2mlir_root), "rev-parse", "HEAD"], text=True).strip(),
                   preparation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_bundle_sha256={name: file_sha256(args.layer_bundle / name) for name in (
                       "inputs.npz", "weights.safetensors", "golden.npy", "capture_receipt.json")},
                   shape=list(golden.shape), original_layer_golden_checked=True,
                   relative_error_to_f32=(proxy-golden).norm().item()/golden.norm().item(),
                   files_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in args.out_dir.iterdir() if path.is_file()})
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
