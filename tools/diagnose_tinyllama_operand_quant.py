"""Compare a captured TinyLlama stage with PyTorch MX operand fake quantization.

This measures the FP4/FP8 placement policy on exact captured weights and input.
PyTorch's BF16 matrix arithmetic is not the Gemmini MX mesh, so the result is
an accuracy diagnostic and must never be accepted as an accelerator oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def diagnose(stage: Path, model2mlir_root: Path, out_dir: Path, *,
             layers: int, capacity: int, expected_fp4: int,
             expected_fp8: int, expected_fp8_bmm: int = 0,
             stage_kind: str = "prefill",
             precision_overrides: Path | None = None,
             profile_path: Path | None = None) -> dict:
    if layers <= 0 or capacity <= 0 or expected_fp4 <= 0 or expected_fp8 <= 0:
        raise ValueError("TinyLlama diagnostic requires positive model and site counts")
    if stage_kind not in {"prefill", "decode"}:
        raise ValueError("TinyLlama diagnostic needs prefill or decode stage kind")
    if expected_fp8_bmm < 0:
        raise ValueError("TinyLlama diagnostic needs a nonnegative rotary BMM count")
    stage = stage.resolve(strict=True)
    model2mlir_root = model2mlir_root.resolve(strict=True)
    if out_dir.exists():
        raise ValueError("diagnostic output directory already exists")
    receipt = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("weights.safetensors", "inputs.npz", "golden.npy", "model.mlir"):
        artifact = receipt["artifacts"][name]
        path = stage / name
        if path.stat().st_size != artifact["bytes"] or _sha(path) != artifact["sha256"]:
            raise ValueError(f"captured TinyLlama {name} differs from its receipt")
    promoted_fqns: set[str] = set()
    override_sha = None
    override_path = None
    overrides = {}
    if precision_overrides is not None:
        if profile_path is None:
            raise ValueError("precision overrides need the selected MX profile")
        from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
        from mx_gemmini_support.model_precision_plan import build_model_precision_plan
        from mx_gemmini_support.target_profile import load_profile, profile_sha256
        from tools.build_merlin_stage import _load_precision_overrides

        profile = load_profile(profile_path.resolve(strict=True))
        override_path = precision_overrides.resolve(strict=True)
        overrides = _load_precision_overrides(
            override_path, receipt["artifacts"]["model.mlir"]["sha256"],
            profile_sha256(profile))
        override_sha = _sha(override_path)
        worklist = build_mx_worklist_from_merlin(
            (stage / "model.mlir").read_text(encoding="utf-8"), profile)
        plan = build_model_precision_plan(
            worklist, profile, precision_overrides=overrides)
        promoted_fqns = {site["fqn"] for site in plan["sites"]
                         if site["region_id"] in overrides}
        if len(promoted_fqns) != len(overrides) or None in promoted_fqns:
            raise ValueError("MX precision override does not identify unique TinyLlama Linears")
    sys.path.insert(0, str(model2mlir_root))
    from workloads.tiny_llama.loader import get_model_and_inputs
    from m2m.capture.causal_session import FixedCachePrefillStage, FixedCacheDecodeStep
    from mxq.nn.operand_capture import MXOperandLinear, quantize_mx_operand

    os.environ.update(M2M_LLAMA_LAYERS=str(layers), M2M_SEQ="128",
                      M2M_LLAMA_ATTENTION="eager", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    wrapped, _ = get_model_and_inputs()
    model = (FixedCachePrefillStage(wrapped.lm, capacity) if stage_kind == "prefill"
             else FixedCacheDecodeStep(wrapped.lm, capacity))
    model.load_state_dict(load_file(stage / "weights.safetensors"), strict=True)
    with np.load(stage / "inputs.npz", allow_pickle=False) as inputs:
        arguments = [torch.from_numpy(inputs[f"in{index}"].copy())
                     for index in range(1 if stage_kind == "prefill" else 3)]
    golden = np.load(stage / "golden.npy", allow_pickle=False)
    with torch.no_grad():
        baseline = model(*arguments)[0].float().cpu().numpy()
    if not np.array_equal(baseline, golden):
        raise ValueError("unquantized reconstruction differs from the captured golden")

    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)]
    if len(linears) != expected_fp4 + len(promoted_fqns):
        raise ValueError("TinyLlama Linear coverage differs from the FP4 and FP8 site counts")
    if not promoted_fqns <= {name for name, _ in linears}:
        raise ValueError("MX precision override does not match captured TinyLlama Linears")
    for name, module in linears:
        parent_name, attr = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, attr, MXOperandLinear(
            module, "mxfp8" if name in promoted_fqns else "mxfp4"))

    original_matmul = torch.matmul
    original_tensor_matmul = torch.Tensor.__matmul__
    calls = {"fp8_attention": 0, "fp8_rotary_bmm": 0, "other": 0,
             "other_tensor_matmul": 0}

    def mx_matmul(lhs, rhs, *args, **kwargs):
        if lhs.ndim == rhs.ndim == 4 and lhs.shape[-1] >= 32 and lhs.shape[-1] % 32 == 0:
            calls["fp8_attention"] += 1
            lq, _, _ = quantize_mx_operand(lhs, "mxfp8", -1)
            rq, _, _ = quantize_mx_operand(rhs, "mxfp8", -2)
            return original_matmul(lq.float(), rq.float(), *args, **kwargs).to(torch.bfloat16).to(lhs.dtype)
        calls["other"] += 1
        return original_matmul(lhs, rhs, *args, **kwargs)

    def mx_tensor_matmul(lhs, rhs):
        if (lhs.ndim == rhs.ndim == 3 and tuple(lhs.shape) == (1, 32, 1)
                and rhs.shape[0:2] == (1, 1)):
            calls["fp8_rotary_bmm"] += 1
            # The target pads this K=1 contraction to one 32-element MX block.
            # model2MLIR's operand quantizer requires that physical block size.
            lq, _, _ = quantize_mx_operand(torch.nn.functional.pad(lhs, (0, 31)),
                                           "mxfp8", -1)
            rq, _, _ = quantize_mx_operand(torch.nn.functional.pad(rhs, (0, 0, 0, 31)),
                                           "mxfp8", -2)
            return original_tensor_matmul(lq.float(), rq.float()).to(torch.bfloat16).to(lhs.dtype)
        calls["other_tensor_matmul"] += 1
        return original_tensor_matmul(lhs, rhs)

    torch.matmul = mx_matmul
    if expected_fp8_bmm:
        torch.Tensor.__matmul__ = mx_tensor_matmul
    try:
        with torch.no_grad():
            proxy = model(*arguments)[0].float().cpu().numpy()
    finally:
        torch.matmul = original_matmul
        torch.Tensor.__matmul__ = original_tensor_matmul
    if calls != {"fp8_attention": expected_fp8,
                 "fp8_rotary_bmm": expected_fp8_bmm, "other": 0,
                 "other_tensor_matmul": 0}:
        raise ValueError(f"FP8 attention coverage differs: {calls}")
    if override_path is not None and _sha(override_path) != override_sha:
        raise ValueError("MX precision overrides changed during diagnostic")
    out_dir.mkdir(parents=True)
    np.save(out_dir / "baseline.npy", baseline)
    np.save(out_dir / "proxy.npy", proxy)
    revision = subprocess.run(["git", "-C", str(model2mlir_root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    result = {
        "schema": "mx_gemmini.tinyllama_operand_fake_quant.v1",
        "status": "diagnostic_not_mesh_oracle",
        "capture_model_sha256": receipt["artifacts"]["model.mlir"]["sha256"],
        "capture_weights_sha256": receipt["artifacts"]["weights.safetensors"]["sha256"],
        "capture_inputs_sha256": receipt["artifacts"]["inputs.npz"]["sha256"],
        "model2mlir_revision": revision,
        "stage_kind": stage_kind,
        "layers": layers, "capacity": capacity,
        "fp4_linear_sites": expected_fp4,
        "fp8_linear_sites": len(promoted_fqns),
        "precision_overrides": overrides,
        "precision_overrides_sha256": override_sha,
        "fp8_attention_calls": calls["fp8_attention"],
        "fp8_rotary_bmm_calls": calls["fp8_rotary_bmm"],
        "baseline_logits_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "proxy_logits_sha256": hashlib.sha256(proxy.tobytes()).hexdigest(),
        "baseline_top_token": int(baseline.argmax()),
        "proxy_top_token": int(proxy.argmax()),
        "proxy_max_abs_vs_unquantized": float(np.max(np.abs(proxy - baseline))),
        "proxy_mean_abs_vs_unquantized": float(np.mean(np.abs(proxy - baseline))),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--expected-fp4", type=int, required=True)
    parser.add_argument("--expected-fp8", type=int, required=True)
    parser.add_argument("--expected-fp8-bmm", type=int, default=0)
    parser.add_argument("--stage-kind", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--precision-overrides", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.stage, args.model2mlir_root, args.out_dir,
                              layers=args.layers, capacity=args.capacity,
                              expected_fp4=args.expected_fp4,
                              expected_fp8=args.expected_fp8,
                              expected_fp8_bmm=args.expected_fp8_bmm,
                              stage_kind=args.stage_kind,
                              precision_overrides=args.precision_overrides,
                              profile_path=args.profile), sort_keys=True))


if __name__ == "__main__":
    main()
