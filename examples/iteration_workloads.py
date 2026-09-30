"""Independent MX-sized capture candidates; no held-out model weights or inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LinearSeam(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.project = nn.Linear(64, 32)

    def forward(self, tokens):
        return self.project(F.gelu(self.norm(self.expand(tokens))))


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(64, 192)
        self.attn_out = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.up = nn.Linear(64, 128)
        self.down = nn.Linear(128, 64)

    def forward(self, tokens):
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = q.reshape(1, 32, 2, 32).transpose(1, 2).contiguous()
        k = k.reshape(1, 32, 2, 32).transpose(1, 2).contiguous()
        v = v.reshape(1, 32, 2, 32).transpose(1, 2).contiguous()
        scores = (q @ k.transpose(-2, -1).contiguous()) * (32 ** -0.5)
        causal = torch.ones(32, 32, device=tokens.device, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(causal, -10000.0).softmax(dim=-1)
        attention = (weights @ v).transpose(1, 2).reshape(1, 32, 64)
        hidden = self.norm(tokens + self.attn_out(attention))
        return hidden + self.down(F.gelu(self.up(hidden)))


class VisionPatches(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 64, kernel_size=4, stride=4)
        self.mix = nn.Linear(64, 64)
        self.project = nn.Linear(64, 64)

    def forward(self, image):
        tokens = self.patch(image).flatten(2).transpose(1, 2).contiguous()
        return (tokens + self.project(F.gelu(self.mix(tokens)))).mean(dim=1)


class PolicyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(64, 64)
        self.language = nn.Linear(64, 64)
        self.state = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.action = nn.Linear(64, 32)

    def forward(self, image_tokens, text_tokens, state_tokens):
        context = torch.cat((self.vision(image_tokens), self.language(text_tokens)), dim=1)
        query = self.state(state_tokens)
        weights = (query @ context.transpose(-2, -1).contiguous()).softmax(dim=-1)
        fused = weights @ context
        return self.action(self.norm(query + fused))


CASES = {
    "linear_seam": (LinearSeam, ((32, 64),)),
    "decoder_block": (DecoderBlock, ((1, 32, 64),)),
    "vision_patches": (VisionPatches, ((1, 3, 16, 32),)),
    "policy_fusion": (PolicyFusion, ((1, 32, 64), (1, 32, 64), (1, 32, 64))),
}


def make_case(name: str) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """Return a reproducible CPU model and synthetic input for structural capture."""
    cls, shapes = CASES[name]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = cls().eval()
        inputs = tuple(torch.randn(shape) for shape in shapes)
    return model, inputs


def materialize_roster(root: Path, contract: Path, policy: Path) -> Path:
    """Freeze the candidate inputs and original exports in a new artifact root."""
    root.mkdir(parents=True, exist_ok=False)
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "mx_gemmini.iteration_roster_candidate.v1",
        "status": "candidate_not_admitted",
        "source_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "contract_sha256": sha256(contract),
        "policy_sha256": sha256(policy),
        "seed": 0,
        "cases": [],
    }
    for name in CASES:
        model, inputs = make_case(name)
        case_dir = root / name
        case_dir.mkdir()
        state_path = case_dir / "state.pt"
        input_path = case_dir / "inputs.pt"
        export_path = case_dir / "original.pt2"
        torch.save(model.state_dict(), state_path)
        torch.save(inputs, input_path)
        torch.export.save(torch.export.export(model, inputs), export_path)
        manifest["cases"].append({
            "name": name,
            "model_class": type(model).__name__,
            "input_shapes": [list(tensor.shape) for tensor in inputs],
            "artifacts": {
                path.name: sha256(path)
                for path in (state_path, input_path, export_path)
            },
        })
    manifest_path = root / "roster.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    print(materialize_roster(args.output_root, args.contract, args.policy))
