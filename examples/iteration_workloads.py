"""Independent MX-sized capture candidates; no held-out model weights or inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from mx_gemmini_support.m2m_adapter import derive_site_inventory
from mx_gemmini_support.rtl_check import check_sources


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


def tool_identity() -> dict:
    """Fingerprint the code and installed tools used to derive site eligibility."""
    import m2m

    support_root = Path(__file__).resolve().parents[1]
    source_paths = tuple(sorted((support_root / "mx_gemmini_support").rglob("*.py"))) + (
        support_root / "examples/iteration_workloads.py",
        support_root / "examples/select_iteration_workloads.py",
        support_root / "examples/replay_iteration_workloads.py",
    )
    sources = {
        str(path.relative_to(support_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    m2m_root = Path(m2m.__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(m2m_root), *args], text=True).strip()
    if git("status", "--porcelain", "--", "m2m"):
        raise ValueError("model2MLIR source checkout has local changes")
    return {"support_sources": sources, "model2mlir_commit": git("rev-parse", "HEAD"),
            "torchao_version": version("torchao")}


def materialize_roster(root: Path, contract: Path, *, rtl_root: Path, source_record: Path) -> Path:
    """Freeze inputs and derived eligibility before anyone chooses precision."""
    contract_bytes = contract.read_bytes()
    checked = check_sources(rtl_root, contract_bytes, source_record.read_bytes())
    root.mkdir(parents=True, exist_ok=False)
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "mx_gemmini.iteration_roster_candidate.v2",
        "status": "candidate_not_admitted",
        "source_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "tool_identity": tool_identity(),
        "contract_sha256": sha256(contract),
        "rtl_source_record_sha256": sha256(source_record),
        "rtl_source_check": checked,
        "seed": 0,
        "cases": [],
    }
    templates = root / "policy_templates"
    templates.mkdir()
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
        inventory = derive_site_inventory(model, inputs, contract_bytes=contract_bytes)
        inventory_path = case_dir / "site-inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        (templates / f"{name}.yaml").write_text(yaml.safe_dump({
            "schema": "mx_gemmini.quantization_policy.v1",
            "default_format": "host",
            "source_graph_sha256": inventory["source_graph_sha256"],
            "module_overrides": {
                site["site_id"].removeprefix("module:"): "host"
                for site in inventory["sites"] if site["kind"] == "linear"
            },
            "functional_overrides": {
                site["site_id"]: "host"
                for site in inventory["sites"] if site["kind"] == "functional"
            },
        }, sort_keys=True))
        manifest["cases"].append({
            "name": name,
            "model_class": type(model).__name__,
            "input_shapes": [list(tensor.shape) for tensor in inputs],
            "artifacts": {
                path.name: sha256(path)
                for path in (state_path, input_path, export_path, inventory_path)
            },
        })
    manifest_path = root / "roster.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


if __name__ == "__main__":
    # ``python -m`` otherwise defines model classes under ``__main__`` and
    # changes the exported graph identity relative to normal package imports.
    from examples.iteration_workloads import materialize_roster as run_materializer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    print(run_materializer(args.output_root, args.contract,
                           rtl_root=args.rtl_root, source_record=args.sources))
