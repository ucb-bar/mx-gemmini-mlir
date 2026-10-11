"""Capture TinyLlama's first softmax input from its decomposed model2MLIR export."""

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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "full-model-mlir", "profile", "qk-inputs",
                 "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.model2mlir_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _source_closure(root, list((root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("selected model2MLIR source closure differs")
    raw = args.full_model_mlir.read_bytes()
    mlir = (gzip.decompress(raw) if args.full_model_mlir.name.endswith(".gz") else raw).decode()
    worklist = build_model2mlir_worklist(mlir, load_profile(args.profile))
    sites = [site for site in worklist["rank3_batch_matmuls"] if site["fqn"] == FQN]
    if ([site["shape_bmnk"] for site in sites] !=
            [[32, 8, 8, 64], [32, 8, 64, 8]] or
            'prov.region_id = "softmax_0"' not in mlir or
            'prov.aten = "aten._softmax.default"' not in mlir):
        raise ValueError("archived TinyLlama attention graph changed")
    qk_inputs = args.qk_inputs.resolve()
    qk_receipt = json.loads((qk_inputs / "receipt.json").read_text())
    if (qk_receipt["full_model_mlir_sha256"] != worklist["source_mlir_sha256"] or
            qk_receipt["model2mlir_source_closure_sha256"] != MODEL2MLIR_SOURCE_CLOSURE or
            qk_receipt["fqn"] != FQN or qk_receipt["region_id"] != sites[0]["region_id"] or
            qk_receipt["seed"] != SEED):
        raise ValueError("QK input receipt differs from selected full model")

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
            selected = node.op == "call_function" and node.target == torch.ops.aten._softmax.default
            if selected:
                operands, _ = self.fetch_args_kwargs_from_env(node)
                stack = node.meta.get("nn_module_stack") or {}
                fqn = list(stack.values())[-1][0] if stack else None
            output = super().run_node(node)
            if selected and fqn == FQN:
                captured.append((node.name, operands[0].detach().float().cpu().numpy().copy(),
                                 output.detach().float().cpu().numpy().copy()))
            return output

    with torch.no_grad():
        Tap(exported.module()).run(*model_inputs)
    if len(captured) != 1:
        raise ValueError("TinyLlama first attention layer needs one softmax")
    node_name, pre_softmax, probabilities = captured[0]
    if pre_softmax.shape != (1, 32, 8, 8) or probabilities.shape != pre_softmax.shape:
        raise ValueError("decomposed softmax shape differs from the full model")
    pre = np.ascontiguousarray(pre_softmax[0, 0], dtype=np.float32)
    prob = np.ascontiguousarray(probabilities[0, 0], dtype=np.float32)
    q = np.load(qk_inputs / "activation.npy", allow_pickle=False)
    key = np.load(qk_inputs / "weight.npy", allow_pickle=False)
    if (q.shape != (8, 64) or key.shape != (64, 8) or
            _sha(q.tobytes()) != qk_receipt["activation_sha256"] or
            _sha(key.tobytes()) != qk_receipt["weight_sha256"]):
        raise ValueError("QK float arrays differ from their captured receipt")
    masked = np.triu(np.ones((8, 8), dtype=bool), 1)
    unmasked = ~masked
    f32_qk = q @ key
    if (not np.all(pre[masked] < -1e20) or
            not np.allclose(pre[unmasked], (f32_qk * np.float32(0.125))[unmasked],
                            rtol=1e-5, atol=1e-6) or
            not np.all(prob[masked] == 0) or
            not np.allclose(prob.sum(axis=1), 1, rtol=1e-5, atol=1e-5)):
        raise ValueError("TinyLlama QK scale, causal mask, or softmax differs")
    out.mkdir(parents=True)
    np.save(out / "pre_softmax.npy", pre)
    np.save(out / "probability.npy", prob)
    receipt = {
        "schema": "mx_gemmini.tinyllama_attention_softmax_capture.v1",
        "scope": "first decoder attention layer, head zero, reduced random-weight model",
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "full_model_mlir_sha256": worklist["source_mlir_sha256"],
        "qk_region_id": sites[0]["region_id"],
        "pv_region_id": sites[1]["region_id"],
        "softmax_region_id": "softmax_0", "fqn": FQN,
        "fx_node": node_name, "seed": SEED,
        "attention_scale": 0.125,
        "mask": "strict_causal_upper_triangle_negative_infinity",
        "masked_values": int(masked.sum()),
        "qk_input_sha256": qk_receipt["activation_sha256"],
        "qk_weight_sha256": qk_receipt["weight_sha256"],
        "pre_softmax_sha256": _sha(pre.tobytes()),
        "probability_sha256": _sha(prob.tobytes()),
        "pre_softmax_npy_sha256": _sha((out / "pre_softmax.npy").read_bytes()),
        "probability_npy_sha256": _sha((out / "probability.npy").read_bytes()),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
