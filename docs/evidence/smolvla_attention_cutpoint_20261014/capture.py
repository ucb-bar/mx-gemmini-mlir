"""Recreate the first SmolVLA flow-attention cutpoint from model2MLIR inputs.

Requires a model2MLIR checkout, LeRobot, PyTorch, NumPy, and the pinned
checkpoint in the local Hugging Face cache. No model is downloaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class AttentionTap(TorchDispatchMode):
    def __init__(self) -> None:
        self.data: dict[str, np.ndarray] = {}
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.bmm.default and len(args) >= 2:
            left, right = args[:2]
            pair = (tuple(left.shape), tuple(right.shape))
            if pair == ((15, 50, 64), (15, 64, 163)) and "q" not in self.data:
                with torch._C._DisableTorchDispatch():
                    self.data["q"] = left.detach().cpu().numpy().copy()
                    self.data["kt"] = right.detach().cpu().numpy().copy()
            if pair == ((15, 50, 163), (15, 163, 64)) and "p" not in self.data:
                with torch._C._DisableTorchDispatch():
                    self.data["p"] = left.detach().cpu().numpy().copy()
                    self.data["v"] = right.detach().cpu().numpy().copy()
        if func == torch.ops.aten.where.self and len(args) >= 3:
            mask = args[0]
            if tuple(mask.shape) == (1, 1, 50, 163) and "mask" not in self.data:
                with torch._C._DisableTorchDispatch():
                    self.data["mask"] = mask.detach().cpu().numpy().copy()
        return func(*args, **(kwargs or {}))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir", type=Path, required=True)
    parser.add_argument("--flow-stage", type=Path, required=True,
                        help="Captured smolvla_session/stages/flow_denoise directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("cutpoint.npz"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.model2mlir / "workloads/smolvla"))
    from loader import (_MODEL_ID, _MODEL_REVISION, _load_pretrained_config,
                        SmolVLAFlowSession)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(2)
    config = _load_pretrained_config()
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(
        _MODEL_ID, config=config, revision=_MODEL_REVISION,
        local_files_only=True, strict=True)
    flow = SmolVLAFlowSession(policy.model.cpu().eval()).eval()
    input_path = args.flow_stage / "inputs.npz"
    golden_path = args.flow_stage / "session_goldens.npz"
    with np.load(input_path, allow_pickle=False) as arrays:
        inputs = tuple(torch.from_numpy(arrays[f"in{i}"].copy()) for i in range(4))
    tap = AttentionTap()
    with torch.no_grad(), tap:
        observed = flow(*inputs)[0].detach().cpu().numpy().copy()
    with torch.no_grad():
        untapped = flow(*inputs)[0].detach().cpu().numpy().copy()
    with np.load(golden_path, allow_pickle=False) as arrays:
        saved_gold = arrays["actions"][0]
    assert np.array_equal(observed, untapped), "instrumentation changed eager output"
    assert set(tap.data) == {"q", "kt", "p", "v", "mask"}
    gold_rmse = float(np.sqrt(np.mean((observed.astype(np.float64) -
                                       saved_gold.astype(np.float64)) ** 2)))
    assert observed.shape == saved_gold.shape and gold_rmse < 1e-3, gold_rmse
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **tap.data)
    print(json.dumps({
        "checkpoint": f"{_MODEL_ID}@{_MODEL_REVISION}",
        "input_sha256": sha256(input_path),
        "saved_golden_sha256": sha256(golden_path),
        "cutpoint_sha256": sha256(args.out),
        "tap_vs_untapped_rmse": 0.0,
        "eager_vs_saved_gold_rmse": gold_rmse,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
