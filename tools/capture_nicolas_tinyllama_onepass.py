"""Capture BF16 activations for Nicolas's source generator on fixed token IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ids", type=Path, required=True,
                        help="model2MLIR bundle inputs.npz, containing in0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with np.load(args.input_ids, allow_pickle=False) as data:
        ids = data["in0"].reshape(-1).astype(np.int64, copy=True)
    if ids.shape != (160,):
        parser.error("source capture needs exactly 160 token IDs")
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("--out-dir must be fresh")

    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, use_safetensors=True, torch_dtype=torch.bfloat16,
        local_files_only=True).eval()
    values: dict[str, np.ndarray] = {}
    handles = []

    def before(name: str):
        def hook(_module, inputs):
            values[name] = inputs[0].detach()[0].float().cpu().numpy()
        return hook

    def after(name: str):
        def hook(_module, _inputs, result):
            tensor = result[0] if isinstance(result, tuple) else result
            values[name] = tensor.detach()[0].float().cpu().numpy()
        return hook

    def rope(_module, inputs, kwargs):
        pair = kwargs.get("position_embeddings")
        if pair is None and len(inputs) >= 2 and isinstance(inputs[1], tuple):
            pair = inputs[1]
        if pair is None:
            raise RuntimeError("could not capture rotary position embeddings")
        values["rope_cos"] = pair[0].detach()[0].float().cpu().numpy()
        values["rope_sin"] = pair[1].detach()[0].float().cpu().numpy()

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(before(f"h_pre{layer_index}")))
        handles.append(layer.register_forward_hook(after(f"h_out{layer_index}")))
    handles.append(model.model.layers[0].self_attn.register_forward_pre_hook(
        rope, with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(before("h_final")))
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(ids)[None], use_cache=False,
                       return_dict=True).logits[0].float().cpu().numpy()
    for handle in handles:
        handle.remove()

    out.mkdir(parents=True)
    common = {"meta_seq": np.asarray(160), "meta_token_ids": ids}
    for layer_index in range(len(model.model.layers)):
        np.savez(out / f"layer{layer_index}.npz",
                 h_pre=values[f"h_pre{layer_index}"],
                 h_out=values[f"h_out{layer_index}"],
                 rope_cos=values["rope_cos"], rope_sin=values["rope_sin"],
                 **common)
    np.savez(out / "model.npz", embed_out=values["h_pre0"],
             h_final=values["h_final"], logits=logits,
             labels=np.concatenate((ids[1:], ids[-1:])), **common)
    print(f"captured {len(model.model.layers)} layers and {logits.shape} logits in {out}")


if __name__ == "__main__":
    main()
