"""Capture an actual first-layer MLP down-projection and residual boundary.

The prefix runs in source FP32. This section is neither a quantized intermediate
dump nor an autoregressive session, and cannot establish whole-model quality.
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
    if args.rows <= 0 or args.out_dir.exists():
        parser.error("positive rows and fresh output required")
    import numpy as np
    import torch
    from m2m.capture.bundle import write_bundle
    from safetensors import safe_open
    from workloads.tiny_llama.loader import get_model_and_inputs

    from mx_gemmini_support.execution_policy import make_policy
    from mx_gemmini_support.target_profile import load_profile

    stage = args.stage.resolve(strict=True)
    receipt = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("model.mlir", "inputs.npz", "weights.safetensors"):
        if sha(stage / name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("source capture differs: " + name)
    with np.load(stage / "inputs.npz", allow_pickle=False) as data:
        ids = data["in0"].reshape(-1).astype(np.int64)
    if args.rows > ids.size:
        raise ValueError("section exceeds the original input sequence")
    out = args.out_dir.resolve()
    out.mkdir()
    np.save(out / "token_ids.npy", ids)
    os.environ.update(M2M_SEQ=str(ids.size), M2M_LLAMA_TOKEN_IDS=str(out / "token_ids.npy"),
                      M2M_LLAMA_ATTENTION="eager", M2M_LLAMA_LAYERS="",
                      M2M_LLAMA_CHECKPOINT_LAYERS="", M2M_LLAMA_SESSION="",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(2)
    model, inputs = get_model_and_inputs()
    prefix = {name: value for name, value in model.named_parameters()
              if name.startswith("lm.model.layers.0.") or name == "lm.model.embed_tokens.weight"}
    with safe_open(str(stage / "weights.safetensors"), framework="pt", device="cpu") as weights:
        for name, tensor in prefix.items():
            if not torch.equal(tensor.detach().view(torch.int32), weights.get_tensor(name).view(torch.int32)):
                raise ValueError("source prefix checkpoint differs: " + name)
    layer = model.lm.model.layers[0]
    captured = {}

    class PrefixCaptured(Exception):
        pass

    def residual_hook(module, arguments):
        captured["residual"] = arguments[0].detach().clone()

    def matrix_hook(module, arguments):
        captured["activation"] = arguments[0].detach().clone()
        raise PrefixCaptured

    hooks = [layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
             layer.mlp.down_proj.register_forward_pre_hook(matrix_hook)]
    try:
        with torch.no_grad():
            model(*inputs)
    except PrefixCaptured:
        pass
    finally:
        for hook in hooks:
            hook.remove()
    if captured.keys() != {"activation", "residual"}:
        raise ValueError("source prefix did not produce the intended private boundary")

    class Section(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer.mlp.down_proj

        def forward(self, x, residual):
            return residual + self.linear(x)

    section_inputs = tuple(captured[name][:, -args.rows:, :].contiguous()
                           for name in ("activation", "residual"))
    write_bundle(Section().eval(), section_inputs, out / "capture", capture_regions=False)
    policy = make_policy((out / "capture/model.mlir").read_text(), load_profile(args.profile))
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    for name in ("model.mlir", "inputs.npz", "weights.safetensors"):
        if sha(stage / name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("source capture changed during section capture: " + name)
    (out / "source_receipt.json").write_text(json.dumps({
        "schema": "mx_gemmini.source_matrix_residual_section.v1",
        "scope": "fp32_first_layer_prefix_to_down_projection_and_residual_not_full_model",
        "source_capture_receipt_sha256": sha(stage / "capture_receipt.json"),
        "source_weights_sha256": sha(stage / "weights.safetensors"),
        "checked_prefix_parameters": list(prefix),
        "input_shapes": [list(t.shape) for t in section_inputs],
        "section_capture_receipt_sha256": sha(out / "capture/capture_receipt.json"),
    }, indent=2) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
