"""Extract deterministic gate and down-projection operands from small LLaMA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
LOADER_SHA256 = "aa859e6ffb6ec765c372511a97d63aba6205b8a5dd092cb2ac5b36864d9c2c5b"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--all-mlp", action="store_true",
                        help="capture gate, up, and down projections in both layers")
    args = parser.parse_args()
    source = args.model2mlir_root.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    loader = source / "workloads/small_llama/loader.py"
    if _sha(loader.read_bytes()) != LOADER_SHA256:
        raise ValueError("small LLaMA source differs")
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=source, text=True).strip()
        if revision != MODEL2MLIR_REVISION:
            raise ValueError("small LLaMA needs pinned model2MLIR revision")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from workloads.small_llama.loader import get_model_and_inputs

    torch.manual_seed(0)
    model, inputs = get_model_and_inputs()
    if tuple(inputs[0].shape) != (1, 8):
        raise ValueError("small LLaMA input token shape differs")
    hooks = []
    captured = {}
    specs = (
        (("b0.g", 0, "g", "matmul_6", (8, 344, 128)),
         ("b0.u", 0, "u", "matmul_7", (8, 344, 128)),
         ("b0.dn", 0, "dn", "matmul_8", (8, 128, 344)),
         ("b1.g", 1, "g", "matmul_15", (8, 344, 128)),
         ("b1.u", 1, "u", "matmul_16", (8, 344, 128)),
         ("b1.dn", 1, "dn", "matmul_17", (8, 128, 344)))
        if args.all_mlp else
        (("g", 0, "g", "matmul_6", (8, 344, 128)),
         ("dn", 0, "dn", "matmul_8", (8, 128, 344)))
    )
    for name, layer, projection, _, _ in specs:
        module = getattr(model.blocks[layer].mlp, projection)

        def take_input(_, operands, *, site=name):
            captured[site] = operands[0].detach().reshape(8, -1).clone()

        hooks.append(module.register_forward_pre_hook(take_input))
    with torch.no_grad():
        model(*inputs)
    for hook in hooks:
        hook.remove()
    out.mkdir(parents=True)
    np.save(out / "input_ids.npy", inputs[0].numpy())
    sites = {}
    for name, layer, projection, region, shape in specs:
        activation = captured[name].numpy().astype(np.float32)
        weight = getattr(model.blocks[layer].mlp, projection).weight.detach().T.contiguous().numpy().astype(np.float32)
        if (activation.shape, weight.shape) != ((shape[0], shape[2]),
                                                (shape[2], shape[1])):
            raise ValueError(f"small LLaMA {name} projection shape differs")
        np.save(out / f"{name}.activation.npy", activation)
        np.save(out / f"{name}.weight.npy", weight)
        sites[name] = {
            "fqn": f"blocks.{layer}.mlp.{projection}", "region_id": region,
            "source_shape_mnk": list(shape),
            "activation_sha256": _sha(activation.tobytes()),
            "weight_sha256": _sha(weight.tobytes()),
            "activation_npy_sha256": _sha((out / f"{name}.activation.npy").read_bytes()),
            "weight_npy_sha256": _sha((out / f"{name}.weight.npy").read_bytes()),
        }
    manifest = {
        "schema": ("mx_gemmini.small_llama_all_mlp_operands.v1" if args.all_mlp
                   else "mx_gemmini.small_llama_mlp_operands.v1"),
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "loader_sha256": LOADER_SHA256,
        "seed": 0,
        "input_ids_npy_sha256": _sha((out / "input_ids.npy").read_bytes()),
        "sites": sites,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  sort_keys=True) + "\n")
    print("small LLaMA MLP operands captured")


if __name__ == "__main__":
    main()
