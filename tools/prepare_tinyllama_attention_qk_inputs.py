"""Tap head-zero QK operands from the exact decomposed TinyLlama export graph."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile
from tools.compile_mx import _source_closure


MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
SEED = 20261010
FQN = "lm.model.layers.0.self_attn"
REGION = "matmul_4"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--full-model-mlir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.model2mlir_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _source_closure(root, list((root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("selected model2MLIR source closure differs")
    profile = load_profile(args.profile)
    raw = args.full_model_mlir.read_bytes()
    mlir = (gzip.decompress(raw) if args.full_model_mlir.name.endswith(".gz") else raw).decode()
    worklist = build_model2mlir_worklist(mlir, profile)
    site = next((site for site in worklist["rank3_batch_matmuls"]
                 if site["region_id"] == REGION), None)
    if site is None or site["fqn"] != FQN or site["shape_bmnk"] != [32, 8, 8, 64]:
        raise ValueError("archived TinyLlama QK site changed")

    os.environ.update(HF_HUB_OFFLINE="1", M2M_LLAMA_LAYERS="2", M2M_SEQ="8")
    os.environ.pop("M2M_LLAMA_SESSION", None)
    sys.path[:0] = [str(root), str(root / "workloads/tiny_llama")]
    import numpy as np
    import torch
    from torch.fx import Interpreter
    from m2m.capture.torch_export import capture_frontend_artifact
    from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions
    import loader

    torch.manual_seed(SEED)
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
                fqn = list(stack.values())[-1][0] if stack else None
                if fqn == FQN:
                    captured.append((node.name, *(operand.detach().float().cpu().numpy().copy()
                                                  for operand in operands)))
            return super().run_node(node)

    with torch.no_grad():
        Tap(exported.module()).run(*model_inputs)
    if len(captured) != 2:
        raise ValueError("TinyLlama first attention layer needs exactly QK and PV bmm")
    node_name, all_activation, all_weight = captured[0]
    if (all_activation.shape != (32, 8, 64) or
            all_weight.shape != (32, 64, 8)):
        raise ValueError("decomposed QK operands differ from archived model2MLIR site")
    activation = np.ascontiguousarray(all_activation[0], dtype=np.float32)
    weight = np.ascontiguousarray(all_weight[0], dtype=np.float32)
    out.mkdir(parents=True)
    np.save(out / "activation.npy", activation)
    np.save(out / "weight.npy", weight)
    receipt = {
        "schema": "mx_gemmini.model2mlir_tinyllama_qk_inputs.v1",
        "scope": "first TinyLlama decoder QK batched contraction, head zero only",
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "full_model_mlir_sha256": worklist["source_mlir_sha256"],
        "region_id": REGION, "fqn": FQN, "source_shape_bmnk": site["shape_bmnk"],
        "batch_index": 0, "fx_node": node_name, "seed": SEED,
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
