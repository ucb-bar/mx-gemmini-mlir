"""Tap a model's first QK or PV head from its model2MLIR export graph."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile
from tools.compile_mx import _source_closure
from tools.prepare_model_projection_inputs import MODELS, SEED


MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--full-model-mlir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("qk", "pv"), default="qk")
    args = parser.parse_args()
    root, out = args.model2mlir_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _source_closure(root, list((root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("selected model2MLIR source closure differs")
    raw = args.full_model_mlir.read_bytes()
    mlir = (gzip.decompress(raw) if args.full_model_mlir.name.endswith(".gz") else raw).decode()
    worklist = build_model2mlir_worklist(mlir, load_profile(args.profile))
    spec = MODELS[args.model]
    fqn = spec["fqn"].rsplit(".q_proj", 1)[0]
    sites = [s for s in worklist["rank3_batch_matmuls"] if s["fqn"] == fqn]
    if len(sites) != 2 or any(s["source_op"] != "batch_matmul" for s in sites):
        raise ValueError("selected attention module does not have one QK/PV pair")
    site = sites[0 if args.stage == "qk" else 1]
    batch, m, n, k = site["shape_bmnk"]
    if ((args.stage == "qk" and (n > k or k % 32)) or
            (args.stage == "pv" and n < k)):
        raise ValueError("selected attention batch matmul is not an MX candidate")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.update(spec["env"])
    for key in ("M2M_QWEN_SESSION", "M2M_GEMMA_SESSION", "M2M_SMOLVLA_SESSION"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import transformers
    from torch.fx import Interpreter
    from m2m.capture.torch_export import capture_frontend_artifact
    from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions

    torch.manual_seed(SEED)
    if args.model == "gemma2_2b":
        # Match the archived offline smoke graph's installed default config.
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
    artifact = capture_frontend_artifact(
        model, model_inputs, export_decomposition_table=torch_mlir_gap_decompositions())
    if not artifact.validation.valid:
        raise ValueError("decomposed model2MLIR export is invalid")
    exported = artifact.exported_program or artifact.original_exported_program
    if exported is None:
        raise ValueError("model2MLIR did not provide an exported program")
    captured = []

    class Tap(Interpreter):
        def run_node(self, node):
            if node.op == "call_function" and node.target == torch.ops.aten.bmm.default:
                operands, _ = self.fetch_args_kwargs_from_env(node)
                stack = node.meta.get("nn_module_stack") or {}
                node_fqn = list(stack.values())[-1][0] if stack else None
                if node_fqn == fqn:
                    captured.append((node.name, *(operand.detach().float().cpu().numpy().copy()
                                                  for operand in operands)))
            return super().run_node(node)

    with torch.no_grad():
        Tap(exported.module()).run(*model_inputs)
    if len(captured) != 2:
        raise ValueError("decomposed attention module needs exactly QK and PV bmm")
    node_name, all_activation, all_weight = captured[0 if args.stage == "qk" else 1]
    if all_activation.shape != (batch, m, k) or all_weight.shape != (batch, k, n):
        raise ValueError("decomposed attention operands differ from archived model2MLIR site")
    activation = np.ascontiguousarray(all_activation[0], dtype=np.float32)
    weight = np.ascontiguousarray(all_weight[0], dtype=np.float32)
    out.mkdir(parents=True)
    np.save(out / "activation.npy", activation)
    np.save(out / "weight.npy", weight)
    receipt = {
        "schema": f"mx_gemmini.model2mlir_random_{args.stage}_inputs.v1",
        "scope": (f"reduced random-weight model; first attention "
                  f"{args.stage.upper()} batch, head zero only"),
        "model": args.model, "seed": SEED,
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "full_model_mlir_sha256": worklist["source_mlir_sha256"],
        "region_id": site["region_id"], "fqn": fqn,
        "source_shape_bmnk": site["shape_bmnk"], "batch_index": 0,
        "fx_node": node_name, "torch_version": torch.__version__,
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
