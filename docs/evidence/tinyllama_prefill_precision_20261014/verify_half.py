"""Verify archived half-model FP8 operand-proxy controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.compare_tinyllama_session import metrics  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> None:
    base = json.loads((HERE / "receipt.json").read_text())
    receipt = json.loads((HERE / "half_sweep_receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.tinyllama_half_precision_proxy.v1"
    assert receipt["status"] == "diagnostic_not_mesh_oracle"
    assert receipt["base_receipt_sha256"] == sha(HERE / "receipt.json")
    for name, expected in receipt["artifact_sha256"].items():
        assert sha(HERE / name) == expected, name
    reference = np.load(HERE / "reference.npy", allow_pickle=False)
    all_sites = set(json.loads((HERE / "all_fp8_result.json").read_text())
                    ["precision_overrides"])
    promoted = set()
    for name, fp8_count, fp4_count in (("first11", 77, 78),
                                       ("last11_plus_head", 78, 77)):
        result = json.loads((HERE / f"{name}_result.json").read_text())
        manifest = json.loads((HERE / f"{name}_overrides.json").read_text())
        proxy = np.load(HERE / f"{name}_proxy.npy", allow_pickle=False)
        assert result["status"] == receipt["status"]
        assert result["model2mlir_revision"] == base["model2mlir_revision"]
        assert result["capture_model_sha256"] == base["capture_model_sha256"]
        assert result["capture_weights_sha256"] == base["checkpoint_weights_sha256"]
        assert result["capture_inputs_sha256"] == base["capture_inputs_sha256"]
        assert result["baseline_logits_sha256"] == hashlib.sha256(
            reference.tobytes()).hexdigest()
        assert result["proxy_logits_sha256"] == hashlib.sha256(
            proxy.tobytes()).hexdigest()
        assert result["precision_overrides_sha256"] == sha(
            HERE / f"{name}_overrides.json")
        assert result["precision_overrides"] == manifest["regions"]
        assert manifest["source_mlir_sha256"] == base["capture_model_sha256"]
        assert manifest["profile_sha256"] == base["profile_sha256"]
        assert result["fp8_linear_sites"] == fp8_count
        assert result["fp4_linear_sites"] == fp4_count
        assert result["fp8_attention_calls"] == 44
        assert result["fp8_rotary_bmm_calls"] == 1
        region_numbers = {int(region.split("_")[1]) for region in manifest["regions"]}
        assert (all(number <= 99 for number in region_numbers) if name == "first11"
                else all(number >= 100 for number in region_numbers))
        assert not promoted & set(manifest["regions"])
        promoted.update(manifest["regions"])
        observed = metrics(proxy, reference)
        for key, value in observed.items():
            assert abs(receipt["metrics"][name][key] - value) < 1e-12, (name, key)
    assert promoted == all_sites
    print("TinyLlama half-model FP8 proxy archive verified")


if __name__ == "__main__":
    verify()
