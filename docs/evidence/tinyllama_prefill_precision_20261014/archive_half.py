"""Add two capture-bound half-model FP8 operand-proxy controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.compare_tinyllama_session import metrics  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first11", "last11-plus-head"):
        parser.add_argument(f"--{name}-run", type=Path, required=True)
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
    args = parser.parse_args()
    if (HERE / "half_sweep_receipt.json").exists():
        parser.error("half sweep already archived")
    base_path = HERE / "receipt.json"
    base = json.loads(base_path.read_text())
    reference = np.load(HERE / "reference.npy", allow_pickle=False)
    assert base["schema"] == "mx_gemmini.tinyllama_prefill_precision.v1"
    names = {
        "first11": (args.first11_run, args.first11_manifest, 77, 78),
        "last11_plus_head": (args.last11_plus_head_run,
                             args.last11_plus_head_manifest, 78, 77),
    }
    values = {}
    for name, (run, manifest_path, fp8_count, fp4_count) in names.items():
        for source, destination in ((run / "proxy.npy", f"{name}_proxy.npy"),
                                    (run / "result.json", f"{name}_result.json"),
                                    (manifest_path, f"{name}_overrides.json")):
            shutil.copyfile(source, HERE / destination)
        result = json.loads((HERE / f"{name}_result.json").read_text())
        manifest = json.loads((HERE / f"{name}_overrides.json").read_text())
        proxy = np.load(HERE / f"{name}_proxy.npy", allow_pickle=False)
        assert result["status"] == "diagnostic_not_mesh_oracle"
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
        values[name] = metrics(proxy, reference)
    assert (set(json.loads((HERE / "first11_overrides.json").read_text())["regions"])
            | set(json.loads((HERE / "last11_plus_head_overrides.json").read_text())["regions"])
            == set(json.loads((HERE / "all_fp8_result.json").read_text())
                   ["precision_overrides"]))
    assert (set(json.loads((HERE / "first11_overrides.json").read_text())["regions"])
            & set(json.loads((HERE / "last11_plus_head_overrides.json").read_text())["regions"])
            == set())
    files = [f"{name}_{suffix}" for name in names
             for suffix in ("proxy.npy", "result.json", "overrides.json")]
    receipt = {
        "schema": "mx_gemmini.tinyllama_half_precision_proxy.v1",
        "scope": "same captured TinyLlama prefill, two disjoint half-model policies",
        "status": "diagnostic_not_mesh_oracle",
        "base_receipt_sha256": sha(base_path),
        "metrics": values,
        "artifact_sha256": {name: sha(HERE / name) for name in files},
    }
    (HERE / "half_sweep_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
