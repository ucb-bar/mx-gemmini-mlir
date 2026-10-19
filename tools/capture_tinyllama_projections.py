"""Capture actual first-layer projections and their FP32 source activation.

This is a matrix section qualifier, not quantized whole-model quality evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.capture_matched_tinyllama_onepass import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.rows <= 0:
        parser.error("fresh output and positive rows required")
    import numpy as np
    import torch
    from m2m.capture.bundle import write_bundle
    from safetensors import safe_open
    from workloads.tiny_llama.loader import get_model_and_inputs

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    stage = args.stage.resolve(strict=True)
    capture = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("model.mlir", "inputs.npz", "weights.safetensors"):
        if sha(stage / name) != capture["artifacts"][name]["sha256"]:
            raise ValueError("source capture changed: " + name)
    with np.load(stage / "inputs.npz", allow_pickle=False) as data:
        ids = data["in0"].reshape(-1).astype(np.int64)
    if args.rows > ids.size:
        raise ValueError("requested section exceeds source token sequence")
    out = args.out_dir.resolve()
    out.mkdir()
    np.save(out / "token_ids.npy", ids)
    os.environ.update(M2M_SEQ=str(ids.size), M2M_LLAMA_TOKEN_IDS=str(out / "token_ids.npy"),
                      M2M_LLAMA_ATTENTION="eager", M2M_LLAMA_LAYERS="",
                      M2M_LLAMA_CHECKPOINT_LAYERS="", M2M_LLAMA_SESSION="",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(2)
    model, inputs = get_model_and_inputs()
    layer = model.lm.model.layers[0]
    tensors = {"lm.model.embed_tokens.weight": model.lm.model.embed_tokens.weight,
               "lm.model.layers.0.input_layernorm.weight": layer.input_layernorm.weight}
    projections = []
    for name in ("q_proj", "k_proj", "v_proj"):
        projection = getattr(layer.self_attn, name)
        projections.append(projection)
        tensors["lm.model.layers.0.self_attn." + name + ".weight"] = projection.weight
    with safe_open(str(stage / "weights.safetensors"), framework="pt", device="cpu") as weights:
        for name, tensor in tensors.items():
            if not torch.equal(tensor.detach().view(torch.int32), weights.get_tensor(name).view(torch.int32)):
                raise ValueError("source checkpoint tensor differs: " + name)
    with torch.no_grad():
        activation = layer.input_layernorm(model.lm.model.embed_tokens(inputs[0]))
        activation = activation.reshape(-1, activation.shape[-1])[-args.rows:].contiguous()

    class Section(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = torch.nn.ModuleList(projections)

        def forward(self, x):
            return torch.cat(tuple(projection(x) for projection in self.projections), dim=-1)

    write_bundle(Section().eval(), (activation,), out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (out / "source_receipt.json").write_text(json.dumps({
        "schema": "mx_gemmini.source_projection_section.v1",
        "scope": "three_projections_of_fp32_first_layer_activation_not_full_model",
        "source_capture_receipt_sha256": sha(stage / "capture_receipt.json"),
        "source_weights_sha256": sha(stage / "weights.safetensors"),
        "source_inputs_sha256": sha(stage / "inputs.npz"),
        "checked_checkpoint_tensors": list(tensors),
        "activation_shape": list(activation.shape),
        "section_capture_receipt_sha256": sha(out / "capture/capture_receipt.json"),
    }, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
