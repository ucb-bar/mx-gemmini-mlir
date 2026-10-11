"""Capture one real forward's q_proj operands from each reduced smoke model.

The saved arrays are the matmul operands.  Qwen and SmolVLA add bias after the
matmul; that epilogue is excluded from the isolated MX program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from tools.compile_mx import _source_closure


MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
SEED = 20261010
MODELS = {
    "deepseek_r1_distill_qwen_1_5b": {
        "loader": "deepseek_r1_qwen", "env": {"M2M_QWEN_LAYERS": "1", "M2M_SEQ": "4"},
        "fqn": "lm.model.layers.0.self_attn.q_proj", "shape": [4, 1536, 1536],
        "source_op": "addmm", "region_id": "matmul_1",
    },
    "gemma2_2b": {
        "loader": "gemma2_2b", "env": {"M2M_GEMMA_LAYERS": "2", "M2M_SEQ": "4"},
        "fqn": "lm.model.layers.0.self_attn.q_proj", "shape": [4, 2048, 2304],
        "source_op": "matmul", "region_id": "matmul_1",
    },
    "smolvla": {
        "loader": "smolvla", "env": {"M2M_SMOLVLA_PRETRAINED": "0",
                                     "M2M_SMOLVLA_VLM_LAYERS": "1",
                                     "M2M_SMOLVLA_EXPERT_LAYERS": "1",
                                     "M2M_SMOLVLA_IMAGE_SIZE": "64"},
        "fqn": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
        "shape": [16, 768, 768], "source_op": "addmm", "region_id": "matmul_0",
    },
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.model2mlir_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _source_closure(root, list((root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("model loader must come from selected model2MLIR source")
    spec = MODELS[args.model]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.update(spec["env"])
    for key in ("M2M_QWEN_SESSION", "M2M_GEMMA_SESSION", "M2M_SMOLVLA_SESSION"):
        os.environ.pop(key, None)
    import numpy as np
    import torch
    import transformers

    torch.manual_seed(SEED)
    if args.model == "gemma2_2b":
        # Reproduce the archived offline preflight's default installed config.
        original = transformers.AutoConfig.from_pretrained

        def offline_default_config(name, *positional, **keyword):
            if name == "google/gemma-2-2b-it":
                return transformers.Gemma2Config()
            return original(name, *positional, **keyword)

        transformers.AutoConfig.from_pretrained = offline_default_config
    loader_path = root / "workloads" / spec["loader"] / "loader.py"
    module_spec = importlib.util.spec_from_file_location("selected_model_loader", loader_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("model2MLIR loader is unavailable")
    loader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loader)
    model, model_inputs = loader.get_model_and_inputs()
    projection = dict(model.named_modules()).get(spec["fqn"])
    if not isinstance(projection, torch.nn.Linear):
        raise ValueError("model2MLIR q_proj module path differs from archived graph")
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
    m, n, k = spec["shape"]
    if len(tapped) != 1 or tapped[0].shape != (m, k):
        raise ValueError(f"q_proj input differs from archived shape: {[x.shape for x in tapped]}")
    activation = tapped[0]
    weight = projection.weight.detach().T.contiguous().float().cpu().numpy().copy()
    if weight.shape != (k, n):
        raise ValueError("q_proj weight differs from archived shape")
    out.mkdir(parents=True)
    np.save(out / "activation.npy", activation)
    np.save(out / "weight.npy", weight)
    receipt = {
        "schema": "mx_gemmini.random_model_projection_inputs.v1",
        "model": args.model, "scope": "reduced random-weight model forward; layer-0 q_proj matmul operands",
        "source_op": spec["source_op"], "region_id": spec["region_id"],
        "module_fqn": spec["fqn"], "shape_mnk": spec["shape"],
        "excluded_addmm_bias_epilogue": spec["source_op"] == "addmm",
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "seed": SEED, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "activation_sha256": _sha(activation.tobytes()),
        "weight_sha256": _sha(weight.tobytes()),
        "activation_npy_sha256": _sha((out / "activation.npy").read_bytes()),
        "weight_npy_sha256": _sha((out / "weight.npy").read_bytes()),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
