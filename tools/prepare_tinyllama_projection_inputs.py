"""Record actual q_proj operands from the reduced random-weight TinyLlama smoke model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from tools.compile_mx import _source_closure


MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
SEED = 20261010


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.model2mlir_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _source_closure(root, list((root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("TinyLlama loader must come from the selected current model2MLIR source")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["M2M_LLAMA_LAYERS"] = "2"
    os.environ["M2M_SEQ"] = "8"
    os.environ.pop("M2M_LLAMA_SESSION", None)
    sys.path.insert(0, str(root / "workloads/tiny_llama"))
    import loader
    import numpy as np
    import torch
    import transformers

    torch.manual_seed(SEED)
    model, model_inputs = loader.get_model_and_inputs()
    projection = model.lm.model.layers[0].self_attn.q_proj
    tapped = []

    def record(_module, operands):
        value = operands[0].detach()
        tapped.append(value.reshape(-1, value.shape[-1]).float().cpu().numpy().copy())

    handle = projection.register_forward_pre_hook(record)
    try:
        with torch.no_grad():
            model(*model_inputs)
    finally:
        handle.remove()
    if len(tapped) != 1 or tapped[0].shape != (8, 2048):
        raise ValueError("TinyLlama q_proj did not receive one 8x2048 activation")
    activation = tapped[0]
    weight = projection.weight.detach().T.contiguous().float().cpu().numpy().copy()
    if weight.shape != (2048, 2048):
        raise ValueError("TinyLlama q_proj weight shape differs")
    out.mkdir(parents=True)
    np.save(out / "activation.npy", activation)
    np.save(out / "weight.npy", weight)
    receipt = {
        "schema": "mx_gemmini.tinyllama_random_projection_inputs.v1",
        "scope": "2-layer random-weight TinyLlama; one full-K layer-0 q_proj activation and weight",
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "seed": SEED, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "input_ids_sha256": _sha(model_inputs[0].cpu().numpy().tobytes()),
        "activation_sha256": _sha(activation.tobytes()),
        "weight_sha256": _sha(weight.tobytes()),
        "activation_npy_sha256": _sha((out / "activation.npy").read_bytes()),
        "weight_npy_sha256": _sha((out / "weight.npy").read_bytes()),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
