"""Isolate measured rotary-angle error through an otherwise FP32 full forward.

This is a sensitivity experiment, not quantized execution or a mesh oracle.
An exact-angle injection must reproduce every captured FP32 logit bit before
the measured-angle injection is evaluated. No compiler quality gate is changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.capture_matched_tinyllama_onepass import sha
from tools.compare_mx_model_quality import _kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--angles", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists() or args.threads <= 0:
        parser.error("output must be fresh and threads positive")
    import torch
    from safetensors.torch import load_file
    from workloads.tiny_llama.loader import get_model_and_inputs

    stage = args.stage.resolve(strict=True)
    receipt = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("weights.safetensors", "inputs.npz", "golden.npy", "extra.npz"):
        path = stage / name
        if sha(path) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("captured artifact differs: " + name)
    with np.load(stage / "inputs.npz", allow_pickle=False) as data:
        ids = data["in0"].copy()
    with np.load(stage / "extra.npz", allow_pickle=False) as data:
        frequency = data["buf::lm.model.rotary_emb.inv_freq"].copy()
    observed = np.load(args.angles, allow_pickle=False)
    if observed.shape != (len(frequency), ids.shape[1]) or not np.isfinite(observed).all():
        raise ValueError("measured angle shape or values differ from the captured forward")
    out = args.out_dir.resolve()
    out.mkdir()
    np.save(out / "token_ids.npy", ids.reshape(-1))
    os.environ.update(M2M_SEQ=str(ids.shape[1]), M2M_LLAMA_TOKEN_IDS=str(out / "token_ids.npy"),
                      M2M_LLAMA_ATTENTION="eager", M2M_LLAMA_LAYERS="",
                      M2M_LLAMA_CHECKPOINT_LAYERS="", M2M_LLAMA_SESSION="",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(args.threads)
    model, inputs = get_model_and_inputs()
    model.load_state_dict(load_file(stage / "weights.safetensors"), strict=True)
    golden = np.load(stage / "golden.npy", allow_pickle=False)
    rotary = model.lm.model.rotary_emb
    positions = torch.arange(ids.shape[1], dtype=torch.float32)
    exact = torch.from_numpy(frequency)[:, None] * positions[None, :]

    def forward_with_angles(angles):
        phases = angles.T.unsqueeze(0)
        phases = torch.cat((phases, phases), dim=-1)
        cos = phases.cos() * rotary.attention_scaling
        sin = phases.sin() * rotary.attention_scaling

        def replace(module, arguments, keywords, output):
            position_ids = keywords.get("position_ids")
            if position_ids is None and len(arguments) > 1:
                position_ids = arguments[1]
            if position_ids is None or not torch.equal(
                    position_ids.cpu(), positions.to(torch.int64).unsqueeze(0)):
                raise ValueError("rotary positions differ from the measured forward")
            return cos.to(output[0]), sin.to(output[1])

        hook = rotary.register_forward_hook(replace, with_kwargs=True)
        try:
            with torch.no_grad():
                result = model(*inputs).float().cpu().numpy()
        finally:
            hook.remove()
        return result

    control = forward_with_angles(exact)
    if not np.array_equal(control.view(np.uint32), golden.view(np.uint32)):
        raise ValueError("exact-angle control does not reproduce the captured FP32 logits")
    print("exact-angle FP32 control matched every logit bit", flush=True)
    candidate = forward_with_angles(torch.from_numpy(observed))
    rows = candidate.reshape(-1, candidate.shape[-1])
    reference = golden.reshape(rows.shape)
    metrics = {}
    for label, indices in (("all_rows", slice(None)), ("late_33_rows", slice(-33, None))):
        kl = _kl(reference[indices].astype(np.float64), rows[indices].astype(np.float64))
        metrics[label] = {"mean_kl_nats": float(kl.mean()), "worst_kl_nats": float(kl.max()),
                          "positions": len(kl), "top1_agreement": int(np.count_nonzero(
                              rows[indices].argmax(-1) == reference[indices].argmax(-1)))}
    np.save(out / "angle_perturbed_fp32_logits.npy", candidate)
    report = {"schema": "mx_gemmini.rotary_angle_sensitivity.v1",
              "scope": "measured_angles_in_otherwise_fp32_forward_not_quantized_model_execution",
              "control_logits_bitwise_equal": True, "metrics": metrics,
              "capture_receipt_sha256": sha(stage / "capture_receipt.json"),
              "angles_sha256": sha(args.angles), "threads": args.threads,
              "output_sha256": sha(out / "angle_perturbed_fp32_logits.npy")}
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
